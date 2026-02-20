"""
MaAI Encoder: Distilled WavLM Student Model.
Ported from VAP_Nodding_para/custom_maai_enc.py for real-time inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Tuple, List, Dict

from .encoder_components import get_cnn_layer


# ----------------------------
# Helper Functions
# ----------------------------

def lengths_to_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    lengths: (B,) 実長（フレーム数）
    return: (B, Lmax) bool (True=pad, False=valid)
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    range_ = torch.arange(max_len, device=lengths.device)[None, :]  # (1, Lmax)
    mask = range_ >= lengths[:, None]  # (B, Lmax)
    return mask


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """ALiBi 用のヘッドごとの slope を生成（Press et al. 2021）"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest)
        extra = get_slopes_power_of_2(2 * closest)
        slopes += extra[0::2][: n_heads - closest]
    return torch.tensor(slopes)


def build_alibi_bias(seq_len: int, n_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    返り値: (n_heads, seq_len, seq_len)
    bias[h, i, j] = slope[h] * (j - i)
    """
    slopes = _get_alibi_slopes(n_heads).to(device=device, dtype=dtype)
    i = torch.arange(seq_len, device=device)
    j = torch.arange(seq_len, device=device)
    bias_2d = j[None, :] - i[:, None]
    alibi = slopes[:, None, None] * bias_2d[None, :, :]
    return alibi


# ----------------------------
# Model Classes
# ----------------------------

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, rank=0, alpha=1, dropout=0.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_dropout = nn.Dropout(dropout)
            self.reset_lora_parameters()
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def reset_lora_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = super().forward(x)
        if self.rank > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


class ChannelNorm(nn.Module):
    """Channel Normalization (from CPC encoder)."""
    def __init__(self, numFeatures, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean) * torch.rsqrt(cumVar + self.epsilon)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CNNFrontEnd(nn.Module):
    """
    Standard WavLM/HuBERT-compatible 7-layer CNN Frontend.
    Target: 50Hz (20ms stride) given 16kHz input.
    """
    def __init__(
        self,
        in_channels: int = 1,
        cnn_dim: int = 512,
        sample_rate: int = 16000,
        target_hz: int = 50,
    ):
        super().__init__()

        self.layer_configs = [
            (10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2),
        ]

        if target_hz == 25:
            self.layer_configs.append((2, 2))

        total_stride = 1
        for _, s in self.layer_configs:
            total_stride *= s
        self.target_hz = int(round(sample_rate / float(total_stride)))

        self.conv_layers = nn.ModuleList()
        current_in = in_channels
        for k, s in self.layer_configs:
            conv = nn.Conv1d(current_in, cnn_dim, kernel_size=k, stride=s, padding=0)
            norm = ChannelNorm(cnn_dim)
            act = nn.GELU()
            self.conv_layers.append(nn.Sequential(conv, norm, act))
            current_in = cnn_dim

    def forward(self, x: torch.Tensor, lengths_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for layer, (k, s) in zip(self.conv_layers, self.layer_configs):
            pad_left = k - 1
            x = F.pad(x, (pad_left, 0))
            x = layer(x)

        Ls = lengths_samples.clone()
        for k, s in self.layer_configs:
            Ls = torch.floor((Ls - 1) / s + 1).to(dtype=torch.long)

        feat = x.transpose(1, 2).contiguous()
        return feat, Ls


class CausalSelfAttentionALiBi(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_context_len: Optional[int] = None,
        lora_rank: int = 0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_context_len = max_context_len

        self.q_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.k_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.v_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.o_proj = LoRALinear(d_model, d_model, rank=lora_rank)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Pre-compute ALiBi slopes as a non-persistent buffer
        self.register_buffer(
            "_alibi_slopes",
            _get_alibi_slopes(n_heads),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D) new input frames.
            key_padding_mask: (B, L_total) True = pad.
            past_k: (B, H, L_past, d) cached keys from previous calls.
            past_v: (B, H, L_past, d) cached values.
        Returns:
            (output, all_k, all_v)
        """
        B, L, D = x.shape
        H = self.n_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)   # (B, H, L, d)
        k_new = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v_new = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        # Concatenate with cached keys/values
        if past_k is not None:
            k = torch.cat([past_k, k_new], dim=2)  # (B, H, L_total, d)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k = k_new
            v = v_new

        L_total = k.shape[2]
        offset = L_total - L  # number of cached frames

        # Attention scores: (B, H, L, L_total)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        # ALiBi bias for KV-cached attention
        # Query absolute positions: [offset, offset+L), Key positions: [0, L_total)
        slopes = self._alibi_slopes.to(device=x.device, dtype=attn_scores.dtype)  # (H,)
        q_pos = torch.arange(offset, L_total, device=x.device, dtype=attn_scores.dtype)
        k_pos = torch.arange(L_total, device=x.device, dtype=attn_scores.dtype)
        rel = k_pos[None, :] - q_pos[:, None]  # (L, L_total) — negative for past keys
        alibi = slopes[:, None, None] * rel[None, :, :]  # (H, L, L_total)
        attn_scores = attn_scores + alibi.unsqueeze(0)

        # Causal mask (with absolute positions)
        causal_mask = k_pos[None, :] > q_pos[:, None]  # (L, L_total)
        if self.max_context_len is not None and self.max_context_len > 0:
            causal_mask = causal_mask | (k_pos[None, :] < q_pos[:, None] - self.max_context_len)
        attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :L_total], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out, k, v


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_context_len: Optional[int] = None,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionALiBi(
            d_model, n_heads,
            dropout=dropout,
            max_context_len=max_context_len,
            lora_rank=lora_rank,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            LoRALinear(d_model, int(d_model * mlp_ratio), rank=lora_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(int(d_model * mlp_ratio), d_model, rank=lora_rank),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (x, new_k, new_v)"""
        attn_out, new_k, new_v = self.attn(
            self.ln1(x), key_padding_mask=key_padding_mask,
            past_k=past_k, past_v=past_v,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_k, new_v


class SubPixelUpsampler1d(nn.Module):
    """1D Sub-pixel Convolution (PixelShuffle) upsampler."""
    def __init__(self, d_model: int, upsample_factor: int, kernel_size: int = 1):
        super().__init__()
        if upsample_factor <= 0:
            raise ValueError(f"upsample_factor must be > 0, got {upsample_factor}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")

        self.r = int(upsample_factor)
        self.k = int(kernel_size)
        self.conv = nn.Conv1d(d_model, d_model * self.r, kernel_size=self.k, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, C, L), got shape={tuple(x.shape)}")

        if self.k > 1:
            x = F.pad(x, (self.k - 1, 0))
        x = self.conv(x)

        B, Cr, L = x.shape
        r = self.r
        C = Cr // r

        x = x.contiguous().view(B, C, r, L)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, L * r)
        return x


class StudentModel(nn.Module):
    """
    50Hz CNN (WavLM互換) -> (proj) -> Causal Transformer(ALiBi)
    target_hz==50: 全層 50Hz で処理
    target_hz!=50: 最初の3層は 50Hz で処理 -> Conv1d で target_hz にダウンサンプル
                   -> 残りの層を target_hz で処理 -> SubPixelUpsampler で 50Hz に戻す
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        target_hz: int = 10,
        cnn_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_context_len: Optional[int] = None,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_hz = int(target_hz)
        self.loss_hz = 50

        self.frontend = CNNFrontEnd(
            in_channels=1, cnn_dim=cnn_dim,
            sample_rate=sample_rate, target_hz=self.loss_hz,
        )
        self.proj = nn.Linear(cnn_dim, d_model)

        self.downsample_factor: int = 1
        self.downsample: Optional[nn.Conv1d] = None
        self.upsample: Optional[nn.Module] = None
        if self.target_hz != self.loss_hz:
            if self.target_hz <= 0:
                raise ValueError(f"target_hz must be > 0, got {self.target_hz}")
            if self.loss_hz % self.target_hz != 0:
                raise ValueError(f"Only integer ratios are supported. loss_hz={self.loss_hz}, target_hz={self.target_hz}")
            self.downsample_factor = self.loss_hz // self.target_hz
            k = self.downsample_factor
            s = self.downsample_factor
            self.downsample = nn.Conv1d(d_model, d_model, kernel_size=k, stride=s, padding=0)
            self.upsample = SubPixelUpsampler1d(d_model=d_model, upsample_factor=self.downsample_factor, kernel_size=1)

        self.n_layers = int(n_layers)
        self.n_50_layers = self.n_layers if self.target_hz == self.loss_hz else min(3, self.n_layers)
        self.n_low_layers = self.n_layers - self.n_50_layers

        max_context_len_50 = max_context_len
        max_context_len_low = max_context_len
        if max_context_len is not None and max_context_len > 0 and self.target_hz != self.loss_hz:
            max_context_len_50 = int(max_context_len) * int(self.downsample_factor)
            max_context_len_low = int(max_context_len)

        self.blocks_50 = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, max_context_len=max_context_len_50, lora_rank=lora_rank,
            )
            for _ in range(self.n_50_layers)
        ])
        self.blocks_low = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, max_context_len=max_context_len_low, lora_rank=lora_rank,
            )
            for _ in range(self.n_low_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _lengths_causal_stride(self, lengths: torch.Tensor, stride: int) -> torch.Tensor:
        return torch.floor((lengths - 1) / stride + 1).to(dtype=torch.long)

    def forward_cnn(
        self,
        wav: torch.Tensor,
        lengths_samples: torch.Tensor,
        n_skip_frames: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run only CNN frontend + linear projection (no Transformer).

        Args:
            wav: (B, T_samples) raw waveform.
            lengths_samples: (B,) sample lengths.
            n_skip_frames: Number of leading CNN frames to skip (overlap).

        Returns:
            (cnn_features, feat_lengths_50)
            cnn_features: (B, T_50, D) projected 50Hz features.
        """
        feats_50, feat_lengths_50 = self.frontend(wav, lengths_samples)
        x_50 = self.proj(feats_50)
        if n_skip_frames > 0:
            x_50 = x_50[:, n_skip_frames:, :]
        return x_50, feat_lengths_50

    def forward_transformer(
        self,
        x_50: torch.Tensor,
    ) -> torch.Tensor:
        """Run Transformer blocks + LayerNorm on pre-computed CNN features (no KV cache).

        Args:
            x_50: (B, T_50, D) concatenated CNN features at 50Hz.

        Returns:
            x_out: (B, T_50, D) Transformer output at 50Hz.
        """
        x = x_50
        for blk in self.blocks_50:
            x, _, _ = blk(x, key_padding_mask=None, past_k=None, past_v=None)

        if self.downsample is not None and self.n_low_layers > 0:
            k = self.downsample_factor
            x_ds = x.transpose(1, 2)
            x_ds = F.pad(x_ds, (k - 1, 0))
            x_ds = self.downsample(x_ds)
            x = x_ds.transpose(1, 2)
            feat_lengths_50 = torch.tensor([x_50.shape[1]], device=x_50.device)
            feat_lengths_low = self._lengths_causal_stride(feat_lengths_50, k)
            key_padding_mask_low = lengths_to_padding_mask(feat_lengths_low)
            for blk in self.blocks_low:
                x, _, _ = blk(x, key_padding_mask=key_padding_mask_low)

        x = self.norm(x)

        if self.upsample is not None and self.n_low_layers > 0:
            x_up = x.transpose(1, 2)
            x_up = self.upsample(x_up)
            x_up = x_up.transpose(1, 2)
            x = x_up[:, :x_50.shape[1], :]

        return x

    def forward(
        self,
        wav: torch.Tensor,
        lengths_samples: torch.Tensor,
        cache: Optional[Dict[str, list]] = None,
        n_skip_frames: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, list]]]:
        """
        Args:
            wav: (B, T_samples) raw waveform.
            lengths_samples: (B,) sample lengths.
            cache: KV cache dict {"blocks_50": [(k,v), ...]}. None = no cache.
            n_skip_frames: Number of leading CNN frames to skip (overlap frames).
        Returns:
            (x_50, feat_lengths_50, key_padding_mask_50, new_cache)
            When cache is provided, new_cache contains updated KV for each block.
        """
        feats_50, feat_lengths_50 = self.frontend(wav, lengths_samples)
        x_50 = self.proj(feats_50)

        # Skip overlap frames for incremental mode
        if n_skip_frames > 0:
            x_50 = x_50[:, n_skip_frames:, :]

        # --- 50Hz Transformer blocks with KV cache ---
        if cache is not None:
            blk_cache_50 = cache.get("blocks_50", [None] * len(self.blocks_50))
        else:
            blk_cache_50 = [None] * len(self.blocks_50)

        x = x_50
        new_cache_50: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.blocks_50):
            past_kv = blk_cache_50[i]
            pk, pv = past_kv if past_kv is not None else (None, None)
            x, new_k, new_v = blk(x, key_padding_mask=None, past_k=pk, past_v=pv)
            new_cache_50.append((new_k, new_v))

        # --- Low-rate blocks (no KV cache for now; n_low_layers=0 for current model) ---
        if self.downsample is not None and self.n_low_layers > 0:
            k = self.downsample_factor
            x_ds = x.transpose(1, 2)
            x_ds = F.pad(x_ds, (k - 1, 0))
            x_ds = self.downsample(x_ds)
            x = x_ds.transpose(1, 2)
            feat_lengths_low = self._lengths_causal_stride(feat_lengths_50, k)
            key_padding_mask_low = lengths_to_padding_mask(feat_lengths_low)
            for blk in self.blocks_low:
                x, _, _ = blk(x, key_padding_mask=key_padding_mask_low)
        else:
            feat_lengths_low = feat_lengths_50

        x = self.norm(x)

        if self.upsample is not None and self.n_low_layers > 0:
            x_up = x.transpose(1, 2)
            x_up = self.upsample(x_up)
            x_up = x_up.transpose(1, 2)
            L50_max = int(feat_lengths_50.max().item())
            x_50 = x_up[:, :L50_max, :]
        else:
            x_50 = x

        new_cache = {"blocks_50": new_cache_50}

        return x_50, feat_lengths_50, None, new_cache


# ----------------------------
# EncoderMaai
# ----------------------------

class EncoderMaai(nn.Module):
    """
    Encoder: waveform -> h
    Uses the StudentModel distilled from WavLM.
    """

    def __init__(self, checkpoint_path: str, freeze: bool = True, lim_context_sec: float = -1, frame_hz: int = 50, lora_rank: int = 0):
        super().__init__()
        self.sample_rate = 16000
        self.frame_hz = frame_hz
        self.lim_context_sec = lim_context_sec
        self.lora_rank = lora_rank

        # If there is no checkpoint, try to find the lowest "val_loss" file in the directory
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            print("Searching for matching checkpoint files...")
            list_dir_and_val = []
            dir_path = os.path.dirname(checkpoint_path)
            import zipfile
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for fname in os.listdir(dir_path):
                    if "val_loss=" in fname and fname.endswith(".ckpt"):
                        try:
                            val_loss_str = fname.split("val_loss=")[1].split(".ckpt")[0]
                            val_loss = float(val_loss_str)
                            full_path = os.path.join(dir_path, fname)
                            if not os.access(full_path, os.R_OK):
                                continue
                            try:
                                with zipfile.ZipFile(full_path, 'r') as zf:
                                    zf.namelist()
                            except (zipfile.BadZipFile, Exception):
                                continue
                            list_dir_and_val.append((full_path, val_loss))
                        except Exception:
                            continue
            if len(list_dir_and_val) == 0:
                raise FileNotFoundError(f"No matching checkpoint files found in directory: {dir_path}")
            list_dir_and_val.sort(key=lambda x: x[1])
            checkpoint_path = list_dir_and_val[0][0]
            print(f"Selected checkpoint file: {checkpoint_path}")

        self.model = self._load_model(checkpoint_path)
        self.output_dim = self.model.proj.out_features
        self.dim = self.output_dim

        self._setup_downsample()

        if freeze:
            self.freeze()
            self.eval()
            if self.downsample is not None:
                self.downsample.train()
        else:
            self.unfreeze()

    def _setup_downsample(self):
        """Setup downsampling layer if needed to match frame_hz."""
        self.downsample = None
        output_hz = int(getattr(self.model, "loss_hz", getattr(self.model.frontend, "target_hz", 50)))
        if output_hz > self.frame_hz:
            if output_hz % int(self.frame_hz) != 0:
                print(
                    f"Warning: Cannot downsample cleanly from {output_hz}Hz to {self.frame_hz}Hz "
                    f"(non-integer ratio). Downsample is disabled."
                )
            else:
                factor = int(output_hz // int(self.frame_hz))
                self.downsample = get_cnn_layer(
                    dim=self.output_dim,
                    kernel=[factor],
                    stride=[factor],
                    dilation=[1],
                    activation="GELU",
                    mode="cconv",
                )
                print(f"Downsampling from {output_hz}Hz to {self.frame_hz}Hz (factor={factor})")
        elif output_hz != self.frame_hz:
            print(f"Warning: Model output_hz ({output_hz}) does not match requested frame_hz ({self.frame_hz}).")

    def _load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading EncoderMaai from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        hparams = checkpoint.get("hyper_parameters", {})

        sample_rate = hparams.get("sample_rate", 16000)
        target_hz = int(hparams.get("target_hz", 50))
        max_context_len = None
        if self.lim_context_sec is not None and self.lim_context_sec > 0:
            max_context_len = int(round(self.lim_context_sec * target_hz))

        model = StudentModel(
            sample_rate=sample_rate,
            target_hz=target_hz,
            cnn_dim=hparams.get("cnn_dim", 512),
            d_model=hparams.get("d_model", 256),
            n_heads=hparams.get("n_heads", 8),
            n_layers=hparams.get("n_layers", 6),
            mlp_ratio=hparams.get("mlp_ratio", 4.0),
            dropout=hparams.get("dropout", 0.0),
            max_context_len=max_context_len,
            lora_rank=self.lora_rank,
        )

        state_dict = checkpoint.get("state_dict", checkpoint)
        student_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student."):
                student_state_dict[k[8:]] = v
            elif not k.startswith("teacher.") and not k.startswith("proj_to_teacher"):
                student_state_dict[k] = v
        if not student_state_dict:
            student_state_dict = state_dict

        keys = model.load_state_dict(student_state_dict, strict=False)
        print(f"Loaded EncoderMaai weights: {keys}")
        return model

    @classmethod
    def from_state_dict(cls, state_dict: dict, frame_hz: int = 50, lim_context_sec: float = -1, n_heads: int | None = None):
        """
        Build EncoderMaai from a training checkpoint state_dict (without needing
        a separate distillation checkpoint). Infers architecture from tensor shapes.

        Args:
            state_dict: Full model state_dict (keys like 'encoder.model.proj.weight', etc.)
            frame_hz: Target frame rate for inference.
            lim_context_sec: Context limit in seconds (-1 = no limit).
            n_heads: Number of attention heads. If None, inferred as d_model // 32
                     (which may not match the training config).

        Returns:
            EncoderMaai instance with correct architecture (weights NOT loaded yet).
        """
        # Extract encoder-specific keys
        enc_sd = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                enc_sd[k[len("encoder."):]] = v

        # Infer architecture from tensor shapes
        proj_w = enc_sd.get("model.proj.weight")
        if proj_w is None:
            raise ValueError("Cannot find 'encoder.model.proj.weight' in state_dict")
        d_model = proj_w.shape[0]
        cnn_dim = proj_w.shape[1]

        # Count transformer layers
        n_50_layers = 0
        n_low_layers = 0
        for k in enc_sd:
            if k.startswith("model.blocks_50.") and k.endswith(".attn.q_proj.weight"):
                n_50_layers += 1
            elif k.startswith("model.blocks_low.") and k.endswith(".attn.q_proj.weight"):
                n_low_layers += 1
        n_layers = n_50_layers + n_low_layers

        # Count frontend CNN layers to determine target_hz
        n_conv_layers = 0
        for k in enc_sd:
            if k.startswith("model.frontend.conv_layers.") and k.endswith(".0.weight"):
                n_conv_layers += 1
        # 7 layers = 50Hz, 8 layers = 25Hz
        if n_conv_layers == 8:
            frontend_target_hz = 25
        else:
            frontend_target_hz = 50

        # Infer target_hz from downsample existence
        has_student_downsample = "model.downsample.weight" in enc_sd
        if has_student_downsample:
            ds_w = enc_sd["model.downsample.weight"]
            ds_factor = ds_w.shape[2]  # kernel_size = stride = factor
            target_hz = frontend_target_hz // ds_factor if frontend_target_hz > 1 else frontend_target_hz
        else:
            target_hz = frontend_target_hz

        # Infer mlp_ratio
        mlp_key = "model.blocks_50.0.mlp.0.weight"
        if mlp_key in enc_sd:
            mlp_out_dim = enc_sd[mlp_key].shape[0]
            mlp_ratio = mlp_out_dim / d_model
        else:
            mlp_ratio = 4.0

        # Infer n_heads (default heuristic: head_dim=32)
        if n_heads is None:
            n_heads = max(1, d_model // 32)

        # Check for LoRA
        has_lora = any("lora_A" in k for k in enc_sd)
        lora_rank = 0
        if has_lora:
            for k, v in enc_sd.items():
                if "lora_A" in k:
                    lora_rank = v.shape[0]
                    break

        # Compute max_context_len
        max_context_len = None
        if lim_context_sec > 0:
            max_context_len = int(round(lim_context_sec * target_hz))

        print(f"[EncoderMaai.from_state_dict] Inferred architecture:")
        print(f"  d_model={d_model}, cnn_dim={cnn_dim}, n_layers={n_layers}")
        print(f"  n_50_layers={n_50_layers}, n_low_layers={n_low_layers}")
        print(f"  target_hz={target_hz}, mlp_ratio={mlp_ratio}, n_heads={n_heads}")
        print(f"  lora_rank={lora_rank}")

        # Build StudentModel
        model = StudentModel(
            sample_rate=16000,
            target_hz=target_hz,
            cnn_dim=cnn_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=0.0,
            max_context_len=max_context_len,
            lora_rank=lora_rank,
        )

        # Build EncoderMaai without calling __init__ (bypass checkpoint loading)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.sample_rate = 16000
        instance.frame_hz = frame_hz
        instance.lim_context_sec = lim_context_sec
        instance.lora_rank = lora_rank
        instance.model = model
        instance.output_dim = d_model
        instance.dim = d_model
        instance._setup_downsample()

        # Load encoder weights from state_dict
        model_sd = {}
        for k, v in enc_sd.items():
            if k.startswith("model."):
                model_sd[k[len("model."):]] = v
        load_result = instance.model.load_state_dict(model_sd, strict=False)
        print(f"[EncoderMaai.from_state_dict] Loaded encoder model weights: {load_result}")

        # Load downsample weights if present
        if instance.downsample is not None:
            ds_sd = {}
            for k, v in enc_sd.items():
                if k.startswith("downsample."):
                    ds_sd[k[len("downsample."):]] = v
            if ds_sd:
                load_result = instance.downsample.load_state_dict(ds_sd, strict=False)
                print(f"[EncoderMaai.from_state_dict] Loaded downsample weights: {load_result}")

        # Freeze for inference
        instance.freeze()
        instance.eval()

        return instance

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        self.freeze()
        if hasattr(self.model, "blocks_low") and len(self.model.blocks_low) > 0:
            last_block = self.model.blocks_low[-1]
        elif hasattr(self.model, "blocks_50") and len(self.model.blocks_50) > 0:
            last_block = self.model.blocks_50[-1]
        else:
            raise RuntimeError("StudentModel has no Transformer blocks to unfreeze.")
        for p in last_block.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}: last Transformer block only.")

    def forward_cnn(
        self,
        waveform: torch.Tensor,
        n_skip_frames: int = 0,
    ) -> torch.Tensor:
        """Run CNN frontend + projection only (no Transformer, no downsample).

        Args:
            waveform: (B, C, T) or (B, T) raw waveform.
            n_skip_frames: Number of leading CNN frames to skip (overlap).

        Returns:
            cnn_features: (B, T_50, D) projected 50Hz features.
        """
        if waveform.ndim == 3:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0, :]
            else:
                waveform = waveform.mean(dim=1)
        lengths = torch.tensor([waveform.shape[1]] * waveform.shape[0], device=waveform.device)
        cnn_features, _ = self.model.forward_cnn(waveform, lengths, n_skip_frames=n_skip_frames)
        return cnn_features

    def forward_transformer(
        self,
        cnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run Transformer blocks + norm + downsample on pre-computed CNN features.

        Args:
            cnn_features: (B, T_50, D) concatenated 50Hz CNN features.

        Returns:
            z: (B, T_out, D) encoded features at target frame_hz.
        """
        z = self.model.forward_transformer(cnn_features)

        if self.downsample is not None:
            z = self.downsample(z)

        return z

    def forward(
        self,
        waveform,
        only_feature_extractor: int = 0,
        cache: Optional[Dict[str, object]] = None,
        n_skip_frames: Optional[int] = None,
    ):
        """
        Args:
            waveform: (B, C, T) or (B, T) raw waveform for one frame step.
            cache: Encoder KV cache dict. None = first call (no cache).
            n_skip_frames: Number of leading CNN frames to skip (overlap).
                If None, auto-compute: 0 when cache is None, else inferred
                from ``cache["n_skip"]`` (default 1 for back-compat).
        Returns:
            z: (B, T_out, D) encoded features.
            new_cache: Updated encoder cache (or None if cache was not used).
        """
        # waveform: (B, C, T) or (B, T)
        if waveform.ndim == 3:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0, :]
            else:
                waveform = waveform.mean(dim=1)

        lengths = torch.tensor([waveform.shape[1]] * waveform.shape[0], device=waveform.device)

        # Determine n_skip
        if n_skip_frames is not None:
            n_skip = n_skip_frames
        elif cache is None:
            n_skip = 0
        else:
            n_skip = cache.get("n_skip", 1)

        model_cache = cache.get("model") if cache is not None else None
        z, _, _, new_model_cache = self.model(
            waveform, lengths, cache=model_cache, n_skip_frames=n_skip,
        )

        if self.downsample is not None:
            ds_buf = cache.get("ds_buf") if cache is not None else None
            kernel_size = self.downsample[1].kernel_size[0]  # e.g. 5
            # Save last (kernel_size - 1) frames for next chunk's downsample
            new_ds_buf = z[:, -(kernel_size - 1):, :].clone()

            if ds_buf is not None:
                # Prepend cached frames and bypass CConv1d's zero-padding
                z_with_ctx = torch.cat([ds_buf, z], dim=1)  # (B, 4+T, D)
                x = z_with_ctx.transpose(1, 2)              # (B, D, 4+T)
                # Direct Conv1d (no CConv1d causal padding — context is real)
                cconv = self.downsample[1]
                x = F.conv1d(x, cconv.weight, cconv.bias,
                             stride=cconv.stride, dilation=cconv.dilation)
                # LayerNorm → GELU (layers [2] and [3] of Sequential)
                x = self.downsample[2](x)
                x = self.downsample[3](x)
                z = x.transpose(1, 2)                        # (B, T_out, D)
            else:
                # First chunk or full-sequence: CConv1d zero-padding is correct
                z = self.downsample(z)
        else:
            new_ds_buf = None

        new_cache = {"model": new_model_cache}
        if new_ds_buf is not None:
            new_cache["ds_buf"] = new_ds_buf

        return z, new_cache
