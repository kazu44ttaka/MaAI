from dataclasses import dataclass, field
from typing import List

BIN_TIMES: list = [0.2, 0.4, 0.6, 0.8]

@dataclass
class VapConfig:
    sample_rate: int = 16000
    frame_hz: int = 10
    bin_times: List[float] = field(default_factory=lambda: BIN_TIMES)

    # Encoder (training flag)
    encoder_type: str = "cpc"
    wav2vec_type: str = "mms"
    hubert_model: str = "hubert_jp"
    freeze_encoder: int = 1  # stupid but works (--vap_freeze_encoder 1)
    load_pretrained: int = 1  # stupid but works (--vap_load_pretrained 1)
    only_feature_extraction: int = 0

    # GPT
    dim: int = 256
    channel_layers: int = 1
    cross_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    context_limit: int = -1

    context_limit_cpc_sec: float = -1

    # Added Multi-task
    lid_classify: int = 0   # 1...last layer, 2...middle layer
    lid_classify_num_class: int = 3
    lid_classify_adversarial: int = 0
    lang_cond: int = 0

    # For prompt
    # dim_prompt: int = 1792
    dim_prompt: int = 256
    dim_prompt_2: int = 256

    @staticmethod
    def add_argparse_args(parser, fields_added=[]):
        for k, v in VapConfig.__dataclass_fields__.items():
            if k == "bin_times":
                parser.add_argument(
                    f"--vap_{k}", nargs="+", type=float, default=v.default_factory()
                )
            else:
                parser.add_argument(f"--vap_{k}", type=v.type, default=v.default)
            fields_added.append(k)
        return parser, fields_added

    @staticmethod
    def args_to_conf(args):
        return VapConfig(
            **{
                k.replace("vap_", ""): v
                for k, v in vars(args).items()
                if k.startswith("vap_")
            }
        )


@dataclass
class NodParaConfig:
    """Configuration for nodding parameter prediction heads.
    
    Mirrors the nod_* fields from VAP_Nodding_para/vap/model_nod_para.py VapConfig
    so that checkpoints can be loaded correctly.
    """
    # Count head mode: 0=3-class (1, 2, 3+), 1=binary (1 vs 2+)
    nod_count_binary: int = 0
    # Swing-up prediction mode
    nod_swing_up_mode: str = "binary_and_value"  # "binary_and_value" or "continuous"

    # Per-head MLP layer count (0=Linear, 1=default layers, 2+=that many layers)
    nod_head_mlp_count: int = 0
    nod_head_mlp_range: int = 0
    nod_head_mlp_speed: int = 0
    nod_head_mlp_swing_binary: int = 0
    nod_head_mlp_swing_value: int = 0
    nod_head_mlp_hidden: int = 128   # MLP hidden dimension (shared across tasks)
    nod_head_mlp_layers: int = 1     # Default MLP layer count (used when head value is 1)

    # Shared encoder before all parameter heads
    nod_shared_encoder: int = 0       # 0=off, 1=on
    nod_shared_encoder_dim: int = 256

    # Dropout for heads (-1 means use VapConfig.dropout)
    nod_head_dropout: float = -1.0

    # GPT output dropout (applied to all heads, 0=disabled)
    gpt_output_dropout: float = 0.0

    # Main head MLP options (VAD/VAP/BC/NOD)
    main_head_mlp: int = 0           # 0=off, 1=on
    main_head_mlp_hidden: int = 128

    # Detach nod param gradient from GPT backbone
    nod_param_detach: int = 0        # 0=off, 1=on