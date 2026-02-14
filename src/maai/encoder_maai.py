"""
MAAI Encoder for inference.
Ported from VAP_Nodding_para/vap/custom_maai_enc.py
Uses the StudentModel distilled from WavLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Tuple, List
from maai.encoder_components import get_cnn_layer

# ----------------------------
# Helper Functions
# ----------------------------

def compute_conv1d_out_length(L_in: torch.Tensor, kernel: int, stride: int, padding_total: int, dilation: int = 1) -> torch.Tensor:
    # L_out = floor((L_in + padding_total - dilation*(kernel-1) - 1) / stride + 1)
    return torch.floor((L_in + padding_total - dilation * (kernel - 1) - 1) / stride + 1).to(dtype=torch.long)


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    lengths: (B,) actual lengths (number of frames)
    return: (B, Lmax) bool (True=pad, False=valid)
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    range_ = torch.arange(max_len, device=lengths.device)[None, :]  # (1, Lmax)
    mask = range_ >= lengths[:, None]  # (B, Lmax)
    return mask


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    ALiBi slopes per head (Press et al. 2021)
    """
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
    Returns: (n_heads, seq_len, seq_len)
    bias[h, i, j] = slope[h] * (j - i)
    """
    slopes = _get_alibi_slopes(n_heads).to(device=device, dtype=dtype)  # (H,)
    i = torch.arange(seq_len, device=device)  # (L,)
    j = torch.arange(seq_len, device=device)  # (L,)
    # (L, L): j - i
    bias_2d = j[None, :] - i[:, None]
    # (H, L, L)
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
    """
    Channel Normalization (from CPC / WavLM frontend)
    """
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
    Total Stride: 320 (5 * 2 * 2 * 2 * 2 * 2 * 2)
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
            # (kernel, stride)
            (10, 5),
            (3, 2),
            (3, 2),
            (3, 2),
            (3, 2),
            (2, 2),
            (2, 2),
        ]

        if sample_rate != 16000 or target_hz != 50:
            print(
                f"Warning: This CNNFrontEnd is optimized for 16kHz -> 50Hz. "
                f"Current settings (sr={sample_rate}, hz={target_hz}) might not match perfectly."
            )

        # 25Hz adaptation
        if target_hz == 25:
            self.layer_configs.append((2, 2))
            print("CNNFrontEnd: Added extra layer for 25Hz adaptation.")
        elif target_hz == 50:
            pass
        else:
            print(f"Warning: CNNFrontEnd is optimized for 50Hz or 25Hz. Target {target_hz}Hz may not match.")

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

        print(f"CNNFrontEnd initialized for {self.target_hz}Hz (WavLM Architecture).")

    def forward(self, x: torch.Tensor, lengths_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T) waveform
        lengths_samples: (B,) waveform lengths [samples]
        return:
          feats: (B, L, C)
          feat_lengths: (B,) L
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        # causal padding (left only)
        for layer, (k, s) in zip(self.conv_layers, self.layer_configs):
            pad_left = k - 1
            x = F.pad(x, (pad_left, 0))
            x = layer(x)

        # compute output lengths
        Ls = lengths_samples.clone()
        for k, s in self.layer_configs:
            # L_out = floor((L_in - 1)/s + 1)
            Ls = torch.floor((Ls - 1) / s + 1).to(dtype=torch.long)

        # (B, L, C)
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, D)
        key_padding_mask: (B, L) bool (True=pad, False=valid)
        """
        B, L, D = x.shape
        H = self.n_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)

        # Scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # (B, H, L, L)

        # ALiBi bias
        alibi = build_alibi_bias(L, H, device=x.device, dtype=attn_scores.dtype)  # (H, L, L)
        attn_scores = attn_scores + alibi[None, :, :, :]

        # Causal mask & window mask
        i_indices = torch.arange(L, device=x.device)[:, None]
        j_indices = torch.arange(L, device=x.device)[None, :]
        mask = j_indices > i_indices

        if self.max_context_len is not None and self.max_context_len > 0:
            mask = mask | (j_indices < i_indices - self.max_context_len)

        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float("-inf"))

        # Padding mask (key side)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out


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
            d_model,
            n_heads,
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class StudentModel(nn.Module):
    """
    50Hz CNN (WavLM-compatible) -> (proj) -> Causal Transformer (ALiBi)
      - target_hz==50: all layers at 50Hz
      - target_hz!=50: first 3 layers at 50Hz -> Conv1d downsample to target_hz
                       -> remaining layers at target_hz -> SubPixelUpsampler back to 50Hz

    Distillation loss is always computed at 50Hz.
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

        # Frontend always produces 50Hz
        self.frontend = CNNFrontEnd(
            in_channels=1,
            cnn_dim=cnn_dim,
            sample_rate=sample_rate,
            target_hz=self.loss_hz,
        )
        self.proj = nn.Linear(cnn_dim, d_model)

        # Optional downsample/upsample between transformer stacks
        self.downsample_factor: int = 1
        self.downsample: Optional[nn.Conv1d] = None
        self.upsample: Optional[nn.Module] = None
        if self.target_hz != self.loss_hz:
            if self.target_hz <= 0:
                raise ValueError(f"target_hz must be > 0, got {self.target_hz}")
            if self.loss_hz % self.target_hz != 0:
                raise ValueError(
                    f"Only integer ratios are supported. loss_hz={self.loss_hz}, target_hz={self.target_hz}"
                )
            self.downsample_factor = self.loss_hz // self.target_hz

            k = self.downsample_factor
            s = self.downsample_factor

            # Causal downsample: left padding + conv1d (kernel=stride) for 50Hz -> target_hz
            self.downsample = nn.Conv1d(d_model, d_model, kernel_size=k, stride=s, padding=0)
            # Sub-pixel upsample (1D PixelShuffle)
            self.upsample = SubPixelUpsampler1d(d_model=d_model, upsample_factor=self.downsample_factor, kernel_size=1)

        # Split Transformer into 50Hz and target_hz parts
        self.n_layers = int(n_layers)
        self.n_50_layers = self.n_layers if self.target_hz == self.loss_hz else min(3, self.n_layers)
        self.n_low_layers = self.n_layers - self.n_50_layers

        # Context frames: interpreted as target_hz frames; 50Hz side is scaled accordingly
        max_context_len_50 = max_context_len
        max_context_len_low = max_context_len
        if max_context_len is not None and max_context_len > 0 and self.target_hz != self.loss_hz:
            max_context_len_50 = int(max_context_len) * int(self.downsample_factor)
            max_context_len_low = int(max_context_len)

        self.blocks_50 = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_context_len=max_context_len_50,
                lora_rank=lora_rank,
            )
            for _ in range(self.n_50_layers)
        ])
        self.blocks_low = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                max_context_len=max_context_len_low,
                lora_rank=lora_rank,
            )
            for _ in range(self.n_low_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _lengths_causal_stride(self, lengths: torch.Tensor, stride: int) -> torch.Tensor:
        # Causal padded conv1d output length: floor((L_in - 1)/stride + 1)
        return torch.floor((lengths - 1) / stride + 1).to(dtype=torch.long)

    def forward(self, wav: torch.Tensor, lengths_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        wav: (B, T)
        lengths_samples: (B,)
        returns:
          h_50: (B, L50, D)  -- always 50Hz
          feat_lengths_50: (B,)
          key_padding_mask_50: (B, L50) bool
        """
        feats_50, feat_lengths_50 = self.frontend(wav, lengths_samples)  # (B, L50, C), (B,)
        x_50 = self.proj(feats_50)                                       # (B, L50, D)

        # 1) 50Hz Transformer
        key_padding_mask_50 = lengths_to_padding_mask(feat_lengths_50)
        x = x_50
        for blk in self.blocks_50:
            x = blk(x, key_padding_mask=key_padding_mask_50)

        # 2) 50Hz -> target_hz (optional, between transformer stacks)
        if self.downsample is not None and self.n_low_layers > 0:
            k = self.downsample_factor
            x_ds = x.transpose(1, 2)  # (B, D, L50)
            x_ds = F.pad(x_ds, (k - 1, 0))
            x_ds = self.downsample(x_ds)  # (B, D, Llow)
            x = x_ds.transpose(1, 2)      # (B, Llow, D)
            feat_lengths_low = self._lengths_causal_stride(feat_lengths_50, k)

            key_padding_mask_low = lengths_to_padding_mask(feat_lengths_low)
            for blk in self.blocks_low:
                x = blk(x, key_padding_mask=key_padding_mask_low)
        else:
            feat_lengths_low = feat_lengths_50

        x = self.norm(x)

        # 3) target_hz -> 50Hz (optional)
        if self.upsample is not None and self.n_low_layers > 0:
            x_up = x.transpose(1, 2)   # (B, D, Llow)
            x_up = self.upsample(x_up) # (B, D, Lup)
            x_up = x_up.transpose(1, 2)  # (B, Lup, D)
            L50_max = int(feat_lengths_50.max().item())
            x_50 = x_up[:, :L50_max, :]
        else:
            x_50 = x

        return x_50, feat_lengths_50, key_padding_mask_50


class SubPixelUpsampler1d(nn.Module):
    """1D Sub-pixel Convolution (PixelShuffle) upsampler.

    Input:  (B, C, L)
    Output: (B, C, L * r)
    """

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
        # x: (B, C, L)
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, C, L), got shape={tuple(x.shape)}")

        # Causal padding (left only). With stride=1, output length becomes exactly L.
        if self.k > 1:
            x = F.pad(x, (self.k - 1, 0))
        x = self.conv(x)  # (B, C*r, L)

        B, Cr, L = x.shape
        r = self.r
        if Cr % r != 0:
            raise RuntimeError(f"Channel dimension {Cr} not divisible by upsample_factor {r}")
        C = Cr // r

        # PixelShuffle 1D: (B, C*r, L) -> (B, C, L*r)
        x = x.contiguous().view(B, C, r, L)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, L * r)
        return x


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
                                print(f"  Skipping (no read permission): {fname}")
                                continue
                            try:
                                with zipfile.ZipFile(full_path, 'r') as zf:
                                    zf.namelist()
                            except (zipfile.BadZipFile, Exception) as ze:
                                print(f"  Skipping (invalid/incomplete file): {fname} ({ze})")
                                continue
                            list_dir_and_val.append((full_path, val_loss))
                        except:
                            continue
            if len(list_dir_and_val) == 0:
                raise FileNotFoundError(f"No matching checkpoint files found in directory: {dir_path}")
            list_dir_and_val.sort(key=lambda x: x[1])
            checkpoint_path = list_dir_and_val[0][0]
            print(f"Selected checkpoint file: {checkpoint_path}")

        # Load model from checkpoint
        self.model = self.load_model(checkpoint_path)
        self.output_dim = self.model.proj.out_features
        self.dim = self.output_dim

        # Check frame rate compatibility
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
                )
                print(f"Downsampling from {output_hz}Hz to {self.frame_hz}Hz (factor={factor})")
        elif output_hz != self.frame_hz:
            print(f"Warning: Model output_hz ({output_hz}) does not match requested frame_hz ({self.frame_hz}).")

        if freeze:
            self.freeze()
            self.eval()
            if self.downsample is not None:
                self.downsample.train()
        else:
            self.unfreeze()

    def load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading EncoderMaai from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Try to get hparams
        hparams = checkpoint.get("hyper_parameters", {})

        sample_rate = hparams.get("sample_rate", 16000)
        target_hz = int(hparams.get("target_hz", 50))
        max_context_len = None
        if self.lim_context_sec is not None and self.lim_context_sec > 0:
            max_context_len = int(round(self.lim_context_sec * target_hz))

        # Create model instance
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

        # Filter state_dict for student model
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

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def get_default_conf(self):
        return {""}

    def unfreeze(self):
        self.freeze()

        # Only the last Transformer layer is trainable
        if hasattr(self.model, "blocks_low") and len(self.model.blocks_low) > 0:
            last_block = self.model.blocks_low[-1]
        elif hasattr(self.model, "blocks_50") and len(self.model.blocks_50) > 0:
            last_block = self.model.blocks_50[-1]
        else:
            raise RuntimeError("StudentModel has no Transformer blocks to unfreeze (blocks_50/blocks_low missing).")

        for p in last_block.parameters():
            p.requires_grad_(True)

        print(f"Trainable {self.__class__.__name__}: last Transformer block only.")

    def forward(self, waveform):
        # waveform: (B, C, T) or (B, T)
        if waveform.ndim == 3:
            # (B, C, T) -> (B, T)
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0, :]
            else:
                waveform = waveform.mean(dim=1)

        lengths = torch.tensor([waveform.shape[1]] * waveform.shape[0], device=waveform.device)
        z, _, _ = self.model(waveform, lengths)

        if self.downsample is not None:
            z = self.downsample(z)

        return z
