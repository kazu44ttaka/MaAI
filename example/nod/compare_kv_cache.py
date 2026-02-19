"""
nod モードの KV キャッシュあり/なしを比較する検証スクリプト:
  D) Maai クラス経由 (本番 process() ループ, KVキャッシュなし)
  E) Maai クラス経由 (本番 process() ループ, KVキャッシュあり)
  F) 20秒 sliding window で full-sequence 推論（学習時相当）

F は各フレームで 20 秒ウィンドウを encode_audio(cache=None)+forward(cache=None) で
処理し、最終フレームの出力を取得。

【DとFが一致しない理由】D は毎フレーム encoder に 1920 サンプル(320 overlap + 1600 chunk)
のみを渡し、e1_full に蓄積して GPT に渡す（インクリメンタル）。一方 F は各フレームで
最大20秒分の音声を一度に encoder に渡す（フルシーケンス）。入力長・符号化方式が根本的に
異なるため、D と F は最初から一致しない（設計上の違いであり、バグではない）。

使い方:
  cd MaAI
  uv run python example/nod/compare_kv_cache.py
"""
import sys, os

# ---- path setup ----
_here = os.path.dirname(os.path.abspath(__file__))
_maai_src = os.path.abspath(os.path.join(_here, "../../src"))
sys.path.insert(0, _maai_src)

import torch
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
WAV_FILE = os.path.join(_here, "..", "wav_sample", "jpn_sumida_16k.wav")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_RATE = 10
CROP_SEC = 40.0
CROP_OFFSET_SEC = 7.5
SAMPLE_RATE = 16000


def load_crop_wav(path, offset_sec, crop_sec):
    data, sr = sf.read(path, dtype='float32')
    assert sr == SAMPLE_RATE
    start = int(offset_sec * sr)
    end = start + int(crop_sec * sr)
    data = data[start:end]
    print(f"Loaded WAV: {path}, samples={len(data)}, duration={len(data)/sr:.2f}s")
    return data


def run_maai_nod(wav_ch1, wav_ch2, device, frame_rate=10, use_kv_cache=False):
    """
    Maai(mode='nod') の process() にチャンクを直接投入して結果を収集。
    F とフレーム境界を一致させるため 1600 サンプル (=0.1s) ずつ投入。
    """
    import queue as _queue
    from maai import Maai
    from maai.input import Base as _InputBase

    class _DummyInput(_InputBase):
        def start(self): pass
        def stop(self): pass
        def get_audio_data(self, q): return None

    dummy1 = _DummyInput()
    dummy2 = _DummyInput()

    maai = Maai(
        mode="nod",
        lang="jp",
        frame_rate=frame_rate,
        context_len_sec=20,
        audio_ch1=dummy1,
        audio_ch2=dummy2,
        device=device,
        use_kv_cache=use_kv_cache,
        print_process_time=False,
    )

    chunk_samples = SAMPLE_RATE // frame_rate  # 1600
    n_samples = min(len(wav_ch1), len(wav_ch2))

    all_p_bc = []
    all_p_nod_short = []
    all_p_nod_long = []
    all_p_nod_long_p = []

    for i in range(0, n_samples, chunk_samples):
        end = i + chunk_samples
        if end > n_samples:
            break
        chunk1 = wav_ch1[i:end]
        chunk2 = wav_ch2[i:end]
        maai.process(chunk1, chunk2)

        while not maai.result_dict_queue.empty():
            try:
                r = maai.result_dict_queue.get_nowait()
            except _queue.Empty:
                break
            all_p_bc.append(float(r['p_bc']))
            all_p_nod_short.append(float(r['p_nod_short']))
            all_p_nod_long.append(float(r['p_nod_long']))
            all_p_nod_long_p.append(float(r['p_nod_long_p']))

    heads = {
        'p_bc': torch.tensor(all_p_bc),
        'p_nod_short': torch.tensor(all_p_nod_short),
        'p_nod_long': torch.tensor(all_p_nod_long),
        'p_nod_long_p': torch.tensor(all_p_nod_long_p),
    }
    return heads


# nod の D/E と同一入力を保証するための padding（Maai process の frame_contxt_padding=320）
NOD_FRAME_CONTXT_PADDING = 320


@torch.no_grad()
def run_sliding_window_nod(wav_ch1, wav_ch2, device, frame_rate=10, context_sec=20.0, use_d_padding=True):
    """
    Approach F: 20秒 sliding window で full-sequence 推論（encode_audio cache=None + forward cache=None）。
    D/E と同じ時刻の最終フレーム出力を取得し、20秒以降の推論が正しいかを検証する。

    use_d_padding: True の場合、D と同様に先頭に 320 サンプルゼロを付与（frame_contxt_padding）。
    """
    from maai import Maai
    from maai.input import Base as _InputBase

    class _DummyInput(_InputBase):
        def start(self): pass
        def stop(self): pass
        def get_audio_data(self, q): return None

    maai = Maai(
        mode="nod",
        lang="jp",
        frame_rate=frame_rate,
        context_len_sec=context_sec,
        audio_ch1=_DummyInput(),
        audio_ch2=_DummyInput(),
        device=device,
        use_kv_cache=False,
        print_process_time=False,
    )
    vap = maai.vap

    context_samples = int(context_sec * SAMPLE_RATE)
    chunk_samples = SAMPLE_RATE // frame_rate
    n_total = min(len(wav_ch1), len(wav_ch2))
    n_frames = n_total // chunk_samples

    all_p_bc = []
    all_p_nod_short = []
    all_p_nod_long = []
    all_p_nod_long_p = []

    print(f"  [Approach F] Processing {n_frames} frames with {context_sec}s sliding window (use_d_padding={use_d_padding})...")
    for fi in range(n_frames):
        end = (fi + 1) * chunk_samples
        start = max(0, end - context_samples)
        w1 = wav_ch1[start:end].astype(np.float32)
        w2 = wav_ch2[start:end].astype(np.float32)

        # D と同一入力: Maai process は常に先頭に frame_contxt_padding=320 ゼロを付与
        if use_d_padding:
            pad = np.zeros(NOD_FRAME_CONTXT_PADDING, dtype=np.float32)
            w1 = np.concatenate([pad, w1])
            w2 = np.concatenate([pad, w2])

        # encode_audio(audio1, audio2): audio1=User(ch1), audio2=ERICA(ch2). 内部で swap あり。
        wav_user_t = torch.from_numpy(w1).float().unsqueeze(0).unsqueeze(0).to(device)
        wav_erica_t = torch.from_numpy(w2).float().unsqueeze(0).unsqueeze(0).to(device)
        x1, x2 = vap.encode_audio(wav_user_t, wav_erica_t)
        out, _ = vap.forward(x1, x2, cache=None)

        all_p_bc.append(float(out['p_bc']))
        all_p_nod_short.append(float(out['p_nod_short']))
        all_p_nod_long.append(float(out['p_nod_long']))
        all_p_nod_long_p.append(float(out['p_nod_long_p']))

        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            t_sec = (fi + 1) * chunk_samples / SAMPLE_RATE
            print(f"    frame {fi+1}/{n_frames} (t={t_sec:.1f}s)  p_bc={out['p_bc']:.4f}  window={start/SAMPLE_RATE:.1f}-{end/SAMPLE_RATE:.1f}s")

    return {
        'p_bc': torch.tensor(all_p_bc),
        'p_nod_short': torch.tensor(all_p_nod_short),
        'p_nod_long': torch.tensor(all_p_nod_long),
        'p_nod_long_p': torch.tensor(all_p_nod_long_p),
    }


def visualize(heads_D, heads_E, wav_user, wav_erica, save_dir, heads_F=None):
    os.makedirs(save_dir, exist_ok=True)

    head_keys = ['p_bc', 'p_nod_short', 'p_nod_long', 'p_nod_long_p']
    n_rows = 1 + len(head_keys)  # waveform + heads

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(22, 3 * n_rows),
        sharex=True,
        gridspec_kw=dict(hspace=0.30, top=0.94, bottom=0.04),
    )

    max_time = len(wav_user) / SAMPLE_RATE

    # Waveform
    max_display = 50000
    step = max(1, len(wav_user) // max_display)
    wav_u = wav_user[::step]
    wav_e = wav_erica[::step]
    t_wav = np.arange(len(wav_u)) / (SAMPLE_RATE / step)

    ax = axes[0]
    ax.plot(t_wav, wav_u, color='red', linewidth=0.4, alpha=0.7, label='User')
    ax.plot(t_wav, wav_e, color='blue', linewidth=0.4, alpha=0.5, label='ERICA')
    ax.set_ylabel('Waveform')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_time])

    for row_idx, key in enumerate(head_keys, start=1):
        ax = axes[row_idx]
        vals_d = heads_D[key].numpy()
        vals_e = heads_E[key].numpy()
        t_d = np.arange(len(vals_d)) / FRAME_RATE
        t_e = np.arange(len(vals_e)) / FRAME_RATE

        ax.plot(t_d, vals_d, color='#9467bd', linewidth=1.0, alpha=0.9, linestyle=':', label='D: no KV')
        ax.plot(t_e, vals_e, color='#ff7f0e', linewidth=1.0, alpha=0.9, linestyle='-.', label='E: KV cache')
        if heads_F is not None and key in heads_F:
            vals_f = heads_F[key].numpy()
            t_f = np.arange(len(vals_f)) / FRAME_RATE
            ax.plot(t_f, vals_f, color='#17becf', linewidth=1.0, alpha=0.9, linestyle='--', label='F: 20s SW')

        min_n = min(len(vals_d), len(vals_e))
        diff = np.abs(vals_d[:min_n] - vals_e[:min_n])
        title_parts = [f"D vs E max_diff={diff.max():.6f}"]
        if heads_F is not None and key in heads_F:
            min_n_df = min(len(vals_d), len(heads_F[key]))
            diff_df = np.abs(vals_d[:min_n_df] - heads_F[key][:min_n_df].numpy())
            title_parts.append(f"D vs F max_diff={diff_df.max():.6f}")
        ax.set_ylabel(key)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{key}  |  {'  |  '.join(title_parts)}", fontsize=9)

    axes[-1].set_xlabel('Time (sec)')

    save_path = os.path.join(save_dir, "nod_kv_cache_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def main():
    print("=" * 80)
    print("nod mode: KV cache comparison (D vs E vs F)")
    print("=" * 80)

    wav_user = load_crop_wav(WAV_FILE, CROP_OFFSET_SEC, CROP_SEC)
    rng = np.random.RandomState(42)
    wav_erica = rng.randn(len(wav_user)).astype(np.float32) * 1e-4

    # D: no KV cache
    print("\n--- Approach D: Maai nod (no KV cache) ---")
    heads_D = run_maai_nod(wav_user, wav_erica, DEVICE, FRAME_RATE, use_kv_cache=False)
    for k, v in heads_D.items():
        print(f"  {k}: frames={len(v)}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    # E: KV cache
    print("\n--- Approach E: Maai nod (WITH KV cache) ---")
    heads_E = run_maai_nod(wav_user, wav_erica, DEVICE, FRAME_RATE, use_kv_cache=True)
    for k, v in heads_E.items():
        print(f"  {k}: frames={len(v)}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    # F: 20秒 sliding window full-sequence
    print("\n--- Approach F: 20s sliding window (full-sequence, 学習時相当) ---")
    heads_F = run_sliding_window_nod(wav_user, wav_erica, DEVICE, FRAME_RATE, context_sec=20.0)
    for k, v in heads_F.items():
        print(f"  {k}: frames={len(v)}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

    # Comparison
    print("\n" + "=" * 80)
    print("Comparison Summary (D vs E vs F)")
    print("=" * 80)
    for key in heads_D:
        min_n = min(len(heads_D[key]), len(heads_E[key]))
        diff = (heads_D[key][:min_n] - heads_E[key][:min_n]).abs()
        print(f"  {key} diff (D vs E): max={diff.max():.6f}, mean={diff.mean():.6f}")

    print("\n  --- D/E vs F (20秒以降の推論検証) ---")
    for key in heads_D:
        min_n_df = min(len(heads_D[key]), len(heads_F[key]))
        diff_df = (heads_D[key][:min_n_df] - heads_F[key][:min_n_df]).abs()
        print(f"  {key} diff (D vs F): max={diff_df.max():.6f}, mean={diff_df.mean():.6f}")
        if min_n_df > 200:
            diff_after20 = (heads_D[key][200:min_n_df] - heads_F[key][200:min_n_df]).abs()
            print(f"    after 20s: max={diff_after20.max():.6f}, mean={diff_after20.mean():.6f}")

    # Diagnostic: 特定フレームでの D vs F
    print("\n  --- Diagnostic: D vs F at frame 199, 249, 299 ---")
    for fi in [199, 249, 299]:
        if fi < min(len(heads_D['p_bc']), len(heads_F['p_bc'])):
            t_sec = (fi + 1) * (SAMPLE_RATE // FRAME_RATE) / SAMPLE_RATE
            d_bc = heads_D['p_bc'][fi].item()
            f_bc = heads_F['p_bc'][fi].item()
            print(f"    frame {fi} (t={t_sec:.1f}s): D p_bc={d_bc:.4f}, F p_bc={f_bc:.4f}, |D-F|={abs(d_bc-f_bc):.4f}")

    # Visualization
    save_dir = os.path.join(_here, "visualizations", "kv_cache_comparison")
    visualize(heads_D, heads_E, wav_user, wav_erica, save_dir, heads_F=heads_F)

    print("\nDone!")


if __name__ == "__main__":
    main()
