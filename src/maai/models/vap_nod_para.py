"""
VapGPT_nod_para: Inference model for nodding parameter prediction.

Architecture mirrors VAP_Nodding_para/vap/model_nod_para.py (VapGPT)
with identical parameter names so that training checkpoints can be loaded
directly via load_state_dict(strict=False).

Key differences from training model:
  - KV cache support in forward (via MaAI's GPT/GPTStereo modules)
  - forward() returns scalar predictions (last frame) instead of full sequences
  - Regression outputs are de-normalized using nod_param_stats
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict

from .config import VapConfig, NodParaConfig
from ..encoder_maai import EncoderMaai
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP


def _resolve_layers(val, default_layers):
    """0 -> None (Linear), 1 -> default_layers, 2+ -> that value"""
    if val == 0:
        return None
    elif val == 1:
        return default_layers
    else:
        return val


def _build_mlp(in_dim, hidden_dim, out_dim, n_layers, dropout):
    """Build MLP with n_layers hidden layers."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class VapGPT_nod_para(nn.Module):
    def __init__(
        self,
        conf: Optional[VapConfig] = None,
        nod_para_conf: Optional[NodParaConfig] = None,
        nod_param_stats: Optional[dict] = None,
    ):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        if nod_para_conf is None:
            nod_para_conf = NodParaConfig()

        self.conf = conf
        self.nod_para_conf = nod_para_conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # Normalization stats for regression outputs (range, speed, swing_up)
        self.nod_param_stats = nod_param_stats if nod_param_stats is not None else {
            'range_mean': 0.0, 'range_std': 1.0,
            'speed_mean': 0.0, 'speed_std': 1.0,
            'swing_up_mean': 0.0, 'swing_up_std': 1.0,
        }

        # -------------------------------------------------------
        # GPT layers (names match training code for state_dict compat)
        # -------------------------------------------------------
        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # -------------------------------------------------------
        # GPT output dropout
        # -------------------------------------------------------
        npc = nod_para_conf
        if npc.gpt_output_dropout > 0:
            self.gpt_output_dropout = nn.Dropout(npc.gpt_output_dropout)
        else:
            self.gpt_output_dropout = None

        # -------------------------------------------------------
        # Main output heads
        # -------------------------------------------------------
        use_main_mlp = npc.main_head_mlp == 1
        mlp_h_main = npc.main_head_mlp_hidden

        if use_main_mlp:
            self.va_classifier = nn.Sequential(
                nn.Linear(conf.dim, mlp_h_main), nn.ReLU(), nn.Dropout(conf.dropout),
                nn.Linear(mlp_h_main, 1)
            )
            self.vap_head = nn.Sequential(
                nn.Linear(conf.dim, mlp_h_main), nn.ReLU(), nn.Dropout(conf.dropout),
                nn.Linear(mlp_h_main, self.objective.n_classes)
            )
            self.bc_head = nn.Sequential(
                nn.Linear(conf.dim, mlp_h_main), nn.ReLU(), nn.Dropout(conf.dropout),
                nn.Linear(mlp_h_main, 1)
            )
            self.bc_detect_head = nn.Sequential(
                nn.Linear(conf.dim, mlp_h_main), nn.ReLU(), nn.Dropout(conf.dropout),
                nn.Linear(mlp_h_main, 1)
            )
            self.nod_head = nn.Sequential(
                nn.Linear(conf.dim, mlp_h_main), nn.ReLU(), nn.Dropout(conf.dropout),
                nn.Linear(mlp_h_main, 1)
            )
        else:
            self.va_classifier = nn.Linear(conf.dim, 1)
            self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)
            self.bc_head = nn.Linear(conf.dim, 1)
            self.bc_detect_head = nn.Linear(conf.dim, 1)
            self.nod_head = nn.Linear(conf.dim, 1)

        # -------------------------------------------------------
        # Nodding parameter prediction heads
        # -------------------------------------------------------
        head_do = npc.nod_head_dropout if npc.nod_head_dropout >= 0 else conf.dropout
        nod_param_input_dim = conf.dim

        # Shared encoder (optional front layer for all param heads)
        self.use_nod_shared_encoder = npc.nod_shared_encoder == 1
        if self.use_nod_shared_encoder:
            self.nod_shared_encoder = nn.Sequential(
                nn.Linear(nod_param_input_dim, npc.nod_shared_encoder_dim),
                nn.ReLU(),
                nn.Dropout(head_do)
            )
            head_input_dim = npc.nod_shared_encoder_dim
        else:
            head_input_dim = nod_param_input_dim

        # Resolve per-head MLP layer counts
        default_mlp_layers = npc.nod_head_mlp_layers
        mlp_layers_count = _resolve_layers(npc.nod_head_mlp_count, default_mlp_layers)
        mlp_layers_range = _resolve_layers(npc.nod_head_mlp_range, default_mlp_layers)
        mlp_layers_speed = _resolve_layers(npc.nod_head_mlp_speed, default_mlp_layers)
        mlp_layers_swing_binary = _resolve_layers(npc.nod_head_mlp_swing_binary, default_mlp_layers)
        mlp_layers_swing_value = _resolve_layers(npc.nod_head_mlp_swing_value, default_mlp_layers)
        mlp_h = npc.nod_head_mlp_hidden

        # Count head
        nod_count_out_dim = 1 if npc.nod_count_binary == 1 else 3
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

    def load_encoder(self, maai_checkpoint: str, frame_hz: int = 10, lim_context_sec: float = -1):
        """Load the MAAI encoder from a pre-trained checkpoint.
        
        Args:
            maai_checkpoint: Path to the MAAI encoder checkpoint (.ckpt)
            frame_hz: Target frame rate for the encoder output
            lim_context_sec: Context limit in seconds (-1 for unlimited)
        """
        self.encoder = EncoderMaai(
            checkpoint_path=maai_checkpoint,
            freeze=True,
            lim_context_sec=lim_context_sec,
            frame_hz=frame_hz,
            lora_rank=0,
        )
        self.encoder = self.encoder.eval()

        # Dimension matching if encoder output != GPT dim
        if self.encoder.output_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(self.encoder.output_dim, self.conf.dim)
            print(f"Added decrease_dimension: {self.encoder.output_dim} -> {self.conf.dim}")

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode two audio channels through the MAAI encoder.
        
        No channel swap (matches training code convention):
          audio1 = listener/ERICA (channel 0)
          audio2 = speaker/user (channel 1)
        
        Args:
            audio1: (B, 1, T) or (B, T) waveform for channel 1
            audio2: (B, 1, T) or (B, T) waveform for channel 2
            
        Returns:
            x1, x2: encoded embeddings (B, T', dim)
        """
        x1 = self.encoder(audio1)
        x2 = self.encoder(audio2)

        # Dimension matching
        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    def _denormalize(self, z_val: float, mean: float, std: float) -> float:
        """Convert z-score normalized value back to original scale."""
        return z_val * std + mean

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for inference with KV cache support.
        
        Args:
            x1: Encoded embeddings for channel 1, (B, T, dim)
            x2: Encoded embeddings for channel 2, (B, T, dim)
            cache: KV cache dict from previous forward pass
            
        Returns:
            Tuple of (predictions_dict, new_cache_dict)
            predictions_dict contains scalar values for the last frame.
        """
        if cache is None:
            cache = {}

        # GPT layers
        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"))
        out = self.ar(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
        }

        # Apply GPT output dropout (inference: should be in eval mode, so no-op)
        gpt_out = out["x"]
        if self.gpt_output_dropout is not None:
            gpt_out = self.gpt_output_dropout(gpt_out)

        # ----- Main heads (last frame, scalar) -----
        p_nod_logit = self.nod_head(gpt_out)
        p_nod = p_nod_logit.sigmoid().to("cpu").tolist()[0][-1][0]

        p_bc_logit = self.bc_head(gpt_out)
        p_bc = p_bc_logit.sigmoid().to("cpu").tolist()[0][-1][0]

        # ----- Nod parameter heads (last frame) -----
        nod_param_input = gpt_out  # (B, T, dim)

        if self.use_nod_shared_encoder:
            nod_param_input = self.nod_shared_encoder(nod_param_input)

        # Count: 3-class softmax or binary sigmoid
        nod_count_logit = self.nod_count_head(nod_param_input)
        if self.nod_para_conf.nod_count_binary == 1:
            # Binary: single output, sigmoid
            nod_count_probs = [nod_count_logit.sigmoid().to("cpu").tolist()[0][-1][0]]
        else:
            # 3-class: softmax
            nod_count_probs = nod_count_logit.softmax(dim=-1).to("cpu").tolist()[0][-1]

        # Range: regression (z-score -> original)
        nod_range_z = self.nod_range_head(nod_param_input).to("cpu").tolist()[0][-1][0]
        nod_range = self._denormalize(
            nod_range_z,
            self.nod_param_stats.get('range_mean', 0.0),
            self.nod_param_stats.get('range_std', 1.0)
        )

        # Speed: regression (z-score -> original)
        nod_speed_z = self.nod_speed_head(nod_param_input).to("cpu").tolist()[0][-1][0]
        nod_speed = self._denormalize(
            nod_speed_z,
            self.nod_param_stats.get('speed_mean', 0.0),
            self.nod_param_stats.get('speed_std', 1.0)
        )

        # Swing-up binary: sigmoid
        swing_up_logit = self.nod_swing_up_binary_head(nod_param_input)
        p_swing_up = swing_up_logit.sigmoid().to("cpu").tolist()[0][-1][0]

        # Swing-up value: regression (z-score -> original)
        swing_up_value_z = self.nod_swing_up_value_head(nod_param_input).to("cpu").tolist()[0][-1][0]
        nod_swing_up_value = self._denormalize(
            swing_up_value_z,
            self.nod_param_stats.get('swing_up_mean', 0.0),
            self.nod_param_stats.get('swing_up_std', 1.0)
        )

        ret = {
            "p_nod": p_nod,
            "p_bc": p_bc,
            "nod_count_probs": nod_count_probs,
            "nod_range": nod_range,
            "nod_speed": nod_speed,
            "p_swing_up": p_swing_up,
            "nod_swing_up_value": nod_swing_up_value,
        }

        return ret, new_cache
