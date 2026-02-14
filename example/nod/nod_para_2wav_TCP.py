#!/usr/bin/env python3
"""
nod_para モードのテストスクリプト（2WAVファイル + TCP送受信）

TcpTransmitter で結果を送信し、TcpReceiver で受信してから ConsoleBar に表示します。
TCP経由のシリアライズ/デシリアライズの動作確認に使用します。

使い方:
    python nod_para_2wav_TCP.py \
        --local_model /path/to/nod_para_checkpoint.ckpt \
        --maai_checkpoint /path/to/maai_encoder.ckpt \
        [--device cpu|cuda] [--frame_rate 10] [--context_len_sec 20] \
        [--ip 127.0.0.1] [--port 50008]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput, NodParaConfig


def parse_args():
    parser = argparse.ArgumentParser(description="nod_para mode test with TCP transmit/receive")
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
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50008)
    return parser.parse_args()


def test():
    args = parse_args()

    wav1 = MaaiInput.Wav(wav_file_path=args.wav1)
    wav2 = MaaiInput.Wav(wav_file_path=args.wav2)

    # TCP transmitter (server)
    output_transmitter = MaaiOutput.TcpTransmitter(
        ip=args.ip, port=args.port, mode="nod_para"
    )
    output_transmitter.start_server()

    # TCP receiver (client)
    output_receiver = MaaiOutput.TcpReceiver(
        ip=args.ip, port=args.port, mode="nod_para"
    )
    output_receiver.start()

    # Console output for display
    output = MaaiOutput.ConsoleBar()

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
        # Send via TCP
        output_transmitter.update(result)
        # Receive via TCP
        result_received = output_receiver.get_result()
        # Display
        output.update(result_received)


if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("\nEnding the test script.")
