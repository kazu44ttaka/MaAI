"""
Example: Nod parameter prediction using a single microphone.

This script demonstrates real-time nodding parameter prediction
(count, range, speed, swing-up) using the MaAI encoder.

Requirements:
    - A trained .ckpt checkpoint file from VAP_Nodding_para training.
    - A microphone connected to your machine.

Usage:
    python nod_para_mic.py --checkpoint /path/to/your_checkpoint.ckpt
"""

import sys
import os
import argparse

# For debugging purposes, you can uncomment the following line to add the src directory to the path.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiOutput


def test(checkpoint_path: str, device: str = "cpu", frame_rate: int = 50):

    # Channel 1: User (microphone input)
    mic = MaaiInput.Wav(wav_file_path="C:/Users/kazu4/git/kyotou-attentive-listening/MaAI/example/wav_sample/jpn_sumida_16k.wav", audio_gain=1.0)
    # mic = MaaiInput.Wav(wav_file_path="C:/Users/kazu4/git/kyotou-attentive-listening/VAP_Nodding_para/20211206_02.wav", audio_gain=1.0)
    # mic = MaaiInput.Mic()

    # Channel 2: System/ERICA (silent in mic-only setup)
    # zero = MaaiInput.Zero(white_noise=True)
    zero = MaaiInput.Zero()

    maai = Maai(
        mode="nod_para",
        lang="jp",
        frame_rate=frame_rate,
        context_len_sec=20,
        audio_ch1=mic,
        audio_ch2=zero,
        device=device,
        local_model=checkpoint_path,
        print_process_time=True,
    )

    # maai = Maai(
    #     mode="nod",
    #     lang="jp",
    #     frame_rate=10,
    #     context_len_sec=20,
    #     audio_ch1=mic,
    #     audio_ch2=zero,
    #     device=device,
    #     force_download=True,
    # )

    output = MaaiOutput.ConsoleBar(bar_type="balance")

    maai.start()

    print("Nod parameter prediction started. Press Ctrl+C to stop.")
    print("-" * 80)

    while True:
        result = maai.get_result()

        output.update(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nod parameter prediction (mic input)")
    _assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join(_assets, "medium2_epoch17-val_nod_all_4.12108.ckpt"), help="Path to the .ckpt checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cpu or cuda")
    parser.add_argument("--frame_rate", type=int, default=10, help="Frame rate (Hz)")
    args = parser.parse_args()

    try:
        test(args.checkpoint, args.device, args.frame_rate)
    except KeyboardInterrupt:
        print("\nEnding the test script.")
