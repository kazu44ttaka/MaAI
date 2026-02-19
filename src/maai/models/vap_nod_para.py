"""
VapGPT_nod_para: Real-time inference model for nodding parameter prediction.
Ported from VAP_Nodding_para/model_nod_para.py with KV-cache support.

Attribute names (ar_channel, ar, etc.) match the training model for direct
state_dict loading from .ckpt files.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .config import VapConfig
from ..encoder_maai import EncoderMaai
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP


class VapGPT_nod_para(nn.Module):
    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # nod_param_stats for denormalization (set externally after loading checkpoint)
        self.nod_param_stats = {
            'range_mean': 0.0, 'range_std': 1.0,
            'speed_mean': 0.0, 'speed_std': 1.0,
            'swing_up_mean': 0.0, 'swing_up_std': 1.0,
        }

        # --- GPT layers (names match training model for state_dict compatibility) ---
        if conf.channel_layers > 0:
            self.ar_channel = GPT(
                dim=conf.dim,
                dff_k=3,
                num_layers=conf.channel_layers,
                num_heads=conf.num_heads,
                dropout=conf.dropout,
                context_limit=conf.context_limit,
            )

        if conf.cross_layers > 0:
            self.ar = GPTStereo(
                dim=conf.dim,
                dff_k=3,
                num_layers=conf.cross_layers,
                num_heads=conf.num_heads,
                dropout=conf.dropout,
                context_limit=conf.context_limit,
            )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # GPT output dropout (no-op in eval mode)
        self.gpt_output_dropout = (
            nn.Dropout(conf.gpt_output_dropout) if conf.gpt_output_dropout > 0 else None
        )

        # --- Output heads ---
        self.va_classifier = nn.Linear(conf.dim, 1)
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)
        self.bc_head = nn.Linear(conf.dim, 1)
        self.bc_detect_head = nn.Linear(conf.dim, 1)
        self.nod_head = nn.Linear(conf.dim, 1)

        # --- Nod parameter prediction heads ---
        nod_param_input_dim = conf.dim
        head_do = conf.nod_head_dropout if conf.nod_head_dropout >= 0 else conf.dropout

        # Shared encoder (optional)
        self.use_nod_shared_encoder = conf.nod_shared_encoder == 1
        if self.use_nod_shared_encoder:
            self.nod_shared_encoder = nn.Sequential(
                nn.Linear(nod_param_input_dim, conf.nod_shared_encoder_dim),
                nn.ReLU(),
                nn.Dropout(head_do),
            )
            head_input_dim = conf.nod_shared_encoder_dim
        else:
            head_input_dim = nod_param_input_dim

        # Helper: resolve MLP layers
        def _resolve_layers(val, default_layers):
            return None if val == 0 else default_layers

        default_mlp_layers = conf.nod_head_mlp_layers
        mlp_layers_count = _resolve_layers(conf.nod_head_mlp_count, default_mlp_layers)
        mlp_layers_range = _resolve_layers(conf.nod_head_mlp_range, default_mlp_layers)
        mlp_layers_speed = _resolve_layers(conf.nod_head_mlp_speed, default_mlp_layers)
        mlp_layers_swing_binary = _resolve_layers(conf.nod_head_mlp_swing_binary, default_mlp_layers)
        mlp_layers_swing_value = _resolve_layers(conf.nod_head_mlp_swing_value, default_mlp_layers)
        mlp_h = conf.nod_head_mlp_hidden

        def _build_mlp(in_dim, hidden_dim, out_dim, n_layers, dropout):
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        # Count head
        nod_count_out_dim = 1 if conf.nod_count_binary == 1 else 3
        if mlp_layers_count is not None:
            self.nod_count_head = _build_mlp(head_input_dim, mlp_h, nod_count_out_dim, mlp_layers_count, head_do)
        else:
            self.nod_count_head = nn.Linear(head_input_dim, nod_count_out_dim)

        # Range head (z-score normalized target)
        if mlp_layers_range is not None:
            self.nod_range_head = _build_mlp(head_input_dim, mlp_h, 1, mlp_layers_range, head_do)
        else:
            self.nod_range_head = nn.Linear(head_input_dim, 1)

        # Speed head (z-score normalized target)
        if mlp_layers_speed is not None:
            self.nod_speed_head = _build_mlp(head_input_dim, mlp_h, 1, mlp_layers_speed, head_do)
        else:
            self.nod_speed_head = nn.Linear(head_input_dim, 1)

        # Swing-up binary head
        if mlp_layers_swing_binary is not None:
            self.nod_swing_up_binary_head = _build_mlp(head_input_dim, mlp_h, 1, mlp_layers_swing_binary, head_do)
        else:
            self.nod_swing_up_binary_head = nn.Linear(head_input_dim, 1)

        # Swing-up value head
        if mlp_layers_swing_value is not None:
            self.nod_swing_up_value_head = _build_mlp(head_input_dim, mlp_h, 1, mlp_layers_swing_value, head_do)
            self.nod_swing_up_continuous_head = _build_mlp(head_input_dim, mlp_h, 1, mlp_layers_swing_value, head_do)
        else:
            self.nod_swing_up_value_head = nn.Linear(head_input_dim, 1)
            self.nod_swing_up_continuous_head = nn.Linear(head_input_dim, 1)

    def load_encoder_from_state_dict(self, state_dict: dict, frame_hz: int = 50, lim_context_sec: float = -1, n_heads: int | None = None):
        """
        Build and attach the EncoderMaai from a training checkpoint state_dict.
        Also sets up decrease_dimension if encoder output dim != GPT dim.
        """
        self.encoder = EncoderMaai.from_state_dict(
            state_dict, frame_hz=frame_hz, lim_context_sec=lim_context_sec, n_heads=n_heads
        )

        # decrease_dimension if encoder output dim != GPT dim
        if self.encoder.output_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(self.encoder.output_dim, self.conf.dim)
            # Try to load weights from state_dict
            dd_weight_key = "decrease_dimension.weight"
            dd_bias_key = "decrease_dimension.bias"
            if dd_weight_key in state_dict:
                self.decrease_dimension.weight = nn.Parameter(state_dict[dd_weight_key])
            if dd_bias_key in state_dict:
                self.decrease_dimension.bias = nn.Parameter(state_dict[dd_bias_key])
            print(f"[VapGPT_nod_para] decrease_dimension: {self.encoder.output_dim} -> {self.conf.dim}")

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor):
        """
        Encode audio for both channels (with channel swap, no KV cache).
        audio_ch1 is the User mic, audio_ch2 is the System/ERICA.
        Training convention: speaker 1 = System (ch1), speaker 2 = User (ch2).
        Channel swap maps: ch1(User) -> x2(speaker2), ch2(System) -> x1(speaker1).

        Returns: (x1, x2)
        """
        x1, _ = self.encoder(audio2)
        x2, _ = self.encoder(audio1)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    def encode_audio_cnn(self, audio1: torch.Tensor, audio2: torch.Tensor,
                         n_skip_frames: int = 0):
        """Run only CNN frontend + projection for both channels (no Transformer).

        Channel swap: ch1(User)->x2(speaker2), ch2(System)->x1(speaker1).

        Returns: (cnn1, cnn2)  — 50Hz CNN features before Transformer.
        """
        cnn1 = self.encoder.forward_cnn(audio2, n_skip_frames=n_skip_frames)
        cnn2 = self.encoder.forward_cnn(audio1, n_skip_frames=n_skip_frames)
        return cnn1, cnn2

    def encode_audio_transformer(self, cnn1: torch.Tensor, cnn2: torch.Tensor):
        """Run Transformer + downsample + dimension reduction on pre-computed CNN features.

        Args:
            cnn1: (B, T_50, D) 50Hz CNN features for speaker 1.
            cnn2: (B, T_50, D) 50Hz CNN features for speaker 2.

        Returns: (x1, x2) — encoded embeddings at target frame_hz.
        """
        x1 = self.encoder.forward_transformer(cnn1)
        x2 = self.encoder.forward_transformer(cnn2)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    @staticmethod
    def denormalize(value: float, mean: float, std: float) -> float:
        """Reverse z-score normalization: value * std + mean"""
        return value * std + mean

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass with KV-cache support for real-time inference.

        Args:
            x1: Encoded audio embeddings for speaker 1. (B, T, D)
            x2: Encoded audio embeddings for speaker 2. (B, T, D)
            cache: Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: (result_dict with scalar predictions, new_cache)
        """
        if cache is None:
            cache = {}

        # GPT layers with KV cache
        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"))
        out = self.ar(
            o1["x"], o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
            "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
            "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
        }

        # GPT output dropout (no-op in eval)
        x_out = out["x"]
        if self.gpt_output_dropout is not None:
            x_out = self.gpt_output_dropout(x_out)

        # --- Auxiliary heads (last frame, scalar) ---
        _bc_logit = self.bc_head(x_out)
        _nod_logit = self.nod_head(x_out)
        p_bc = _bc_logit.sigmoid().to("cpu").tolist()[0][-1][0]
        p_nod = _nod_logit.sigmoid().to("cpu").tolist()[0][-1][0]

        # --- Nod parameter heads ---
        nod_param_input = x_out
        if self.use_nod_shared_encoder:
            nod_param_input = self.nod_shared_encoder(nod_param_input)

        # Extract last frame for all heads
        nod_param_last = nod_param_input[:, -1:, :]  # (B, 1, D)

        # Count
        nod_count_raw = self.nod_count_head(nod_param_last)  # (B, 1, C)
        if self.conf.nod_count_binary == 1:
            nod_count_val = nod_count_raw.sigmoid().to("cpu").tolist()[0][0][0]
        else:
            nod_count_val = nod_count_raw.softmax(dim=-1).to("cpu").tolist()[0][0]

        # Range (denormalized)
        nod_range_z = self.nod_range_head(nod_param_last).to("cpu").item()
        nod_range_val = self.denormalize(
            nod_range_z,
            self.nod_param_stats['range_mean'],
            self.nod_param_stats['range_std'],
        )

        # Speed (denormalized)
        nod_speed_z = self.nod_speed_head(nod_param_last).to("cpu").item()
        nod_speed_val = self.denormalize(
            nod_speed_z,
            self.nod_param_stats['speed_mean'],
            self.nod_param_stats['speed_std'],
        )

        # Swing-up binary
        nod_swing_up_binary_val = self.nod_swing_up_binary_head(nod_param_last).sigmoid().to("cpu").item()

        # Swing-up value (denormalized)
        nod_swing_up_value_z = self.nod_swing_up_value_head(nod_param_last).to("cpu").item()
        nod_swing_up_value_val = self.denormalize(
            nod_swing_up_value_z,
            self.nod_param_stats.get('swing_up_mean', 0.0),
            self.nod_param_stats.get('swing_up_std', 1.0),
        )

        # Swing-up continuous (denormalized)
        nod_swing_up_continuous_z = self.nod_swing_up_continuous_head(nod_param_last).to("cpu").item()
        nod_swing_up_continuous_val = self.denormalize(
            nod_swing_up_continuous_z,
            self.nod_param_stats.get('swing_up_mean', 0.0),
            self.nod_param_stats.get('swing_up_std', 1.0),
        )

        ret = {
            "p_bc": p_bc,
            "p_nod": p_nod,
            "nod_count": nod_count_val,
            "nod_range": nod_range_val,
            "nod_speed": nod_speed_val,
            "nod_swing_up_binary": nod_swing_up_binary_val,
            "nod_swing_up_value": nod_swing_up_value_val,
            "nod_swing_up_continuous": nod_swing_up_continuous_val,
        }

        return ret, new_cache
