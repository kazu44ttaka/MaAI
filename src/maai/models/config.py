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

    cpc_model_pt: str = ""

    maai_model_pt: str = "zoom-tri"
    maai_model_run_name: str = "zoom-tri"

    # Conformer student distilled by maai_encoder/distill_wavlm_student_conf.py
    maai_conf_model_pt: str = ""
    maai_conf_model_run_name: str = ""

    lora_rank: int = 0

    freeze_encoder: int = 1  # stupid but works (--vap_freeze_encoder 1)

    # VAP pretrained
    load_pretrained: int = 1  # stupid but works (--vap_load_pretrained 1)

    load_pretrained_whole: int = 0
    load_pretrained_whole_filename: str = ""
    load_pretrained_whole_runname: str = ""

    pretrained_whole: str = ""
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

    # Nod parameter prediction settings
    nod_count_binary: int = 0  # 0=3クラス(1,2,3+), 1=2値(1回 vs 2回以上)

    # Nod parameter head options（タスクごとのMLP有効化設定）
    # 各ヘッド: 0=Linear, 1=MLP（層数はnod_head_mlp_layersで指定）
    nod_head_mlp_count: int = 0
    nod_head_mlp_range: int = 0
    nod_head_mlp_speed: int = 0
    nod_head_mlp_swing_binary: int = 0
    nod_head_mlp_swing_value: int = 0
    nod_head_mlp_hidden: int = 128  # MLP隠れ層次元（全タスク共通）
    nod_head_mlp_layers: int = 1    # MLP隠れ層数（全タスク共通）
    nod_shared_encoder: int = 0     # 0=off, 1=on 全タスク共有特徴抽出層
    nod_shared_encoder_dim: int = 256
    nod_head_dropout: float = -1.0  # shared encoder内・MLP内のDropout率。-1=dropoutと同じ値を使用

    # GPT層出力に対するDropout（全ヘッド共通、0=無効）
    gpt_output_dropout: float = 0.0

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