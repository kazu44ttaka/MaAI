#!/usr/bin/env python3
"""
nod_para モードのテストスクリプト（2WAVファイル + GuiPlot可視化）

全ての頷きパラメータ（p_nod, p_bc, nod_count_probs, nod_range, nod_speed,
p_swing_up, nod_swing_up_value）をリアルタイムにプロット表示します。

使い方:
    python nod_para_2wav_GuiPlot.py \
        --local_model /path/to/nod_para_checkpoint.ckpt \
        --maai_checkpoint /path/to/maai_encoder.ckpt \
        [--device cpu|cuda] [--frame_rate 10] [--context_len_sec 20]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput, NodParaConfig


def parse_args():
    parser = argparse.ArgumentParser(description="nod_para mode test with GuiPlot visualization")
    parser.add_argument("--local_model", type=str, required=True,
                        help="Path to the nod_para PL checkpoint (.ckpt)")
    parser.add_argument("--maai_checkpoint", type=str, required=True,
                        help="Path to the MAAI encoder checkpoint (.ckpt)")
    parser.add_argument("--wav1", type=str, default="../wav_sample/jpn_inoue_16k.wav",
                        help="Path to the first WAV file (listener)")
    parser.add_argument("--wav2", type=str, default="../wav_sample/jpn_sumida_16k.wav",
                        help="Path to the second WAV file (speaker)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frame_rate", type=int, default=10)
    parser.add_argument("--context_len_sec", type=int, default=20)
    parser.add_argument("--shown_context_sec", type=int, default=10,
                        help="Number of seconds shown in the plot window")
    return parser.parse_args()


def test():
    args = parse_args()

    wav1 = MaaiInput.Wav(wav_file_path=args.wav1)
    wav2 = MaaiInput.Wav(wav_file_path=args.wav2)

    output = MaaiOutput.GuiPlot(
        shown_context_sec=args.shown_context_sec,
        frame_rate=args.frame_rate,
        figsize=(18, 14),
    )

    # NodParaConfig: チェックポイントの学習設定に合わせる
    nod_para_conf = NodParaConfig(
        nod_head_mlp_count=1,
        nod_head_mlp_range=1,
        nod_head_mlp_speed=1,
        nod_head_mlp_swing_binary=1,
        nod_head_mlp_swing_value=0,
        nod_head_mlp_hidden=128,
        nod_shared_encoder=1,
        nod_shared_encoder_dim=128,
        nod_head_dropout=0.3,
        gpt_output_dropout=0.2,
    )

    maai = Maai(
        mode="nod_para",
        lang="jp",
        frame_rate=args.frame_rate,
        context_len_sec=args.context_len_sec,
        audio_ch1=wav1,
        audio_ch2=wav2,
        device=args.device,
        local_model=args.local_model,
        maai_checkpoint=args.maai_checkpoint,
        nod_para_conf=nod_para_conf,
    )

    maai.start()

    while True:
        result = maai.get_result()
        output.update(result)


if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("\nEnding the test script.")
