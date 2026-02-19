"""
6つのアプローチを比較する検証スクリプト:
  A) 訓練コードの forward 相当 — フルシーケンスを一括処理
  B) MaAI の encode_audio + forward で丸ごと処理
  C) MaAI の疑似ストリーミング (KVキャッシュなし、フレームを蓄積して毎回全体を処理)
  D) Maai クラス経由 (本番 process() ループ, KVキャッシュなし)
  E) Maai クラス経由 (本番 process() ループ, KVキャッシュあり)
  F) 訓練コードの forward — 20秒 sliding window で各フレーム処理
     (D と同じウィンドウを学習コードで処理し、最終フレームの p_nod を比較)

使い方:
  cd MaAI
  uv run python example/nod_para/compare_three_approaches.py
"""
import sys, os, time

# ---- path setup ----
_here = os.path.dirname(os.path.abspath(__file__))
_maai_src = os.path.abspath(os.path.join(_here, "../../src"))
_maai_root = os.path.abspath(os.path.join(_here, "../../"))
_assets = os.path.join(_maai_root, "assets")
_repo_root = os.path.abspath(os.path.join(_here, "../../../"))
_vap_root = os.path.join(_repo_root, "VAP_Nodding_para")  # 訓練コードのルート
sys.path.insert(0, _maai_src)
sys.path.insert(0, _vap_root)  # vap パッケージを import 可能にする

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
RUN_ABC = False  # True: A/B/C を実行。False: A/B/C をスキップ（D/E/F のみ）
# CHECKPOINT = os.path.join(_assets, "medium2_epoch17-val_nod_all_4.12108.ckpt")
CHECKPOINT = os.path.join(_assets, "medium_epoch17-val_nod_all_4.35415.ckpt")
WAV_FILE = r"C:\Users\kazu4\git\kyotou-attentive-listening\MaAI\example\wav_sample\jpn_sumida_16k.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_RATE = 10  # inference frame rate (Hz)
CROP_SEC = 40.0  # crop length
CROP_OFFSET_SEC = 7.5  # start offset for cropping
SAMPLE_RATE = 16000

USE_ANCHOR_FRAMES = True

# ===========================================================================
# 1. Load and crop WAV
# ===========================================================================
def load_crop_wav(wav_path, offset_sec, crop_sec, sr=16000):
    data, file_sr = sf.read(wav_path, dtype="float32")
    if file_sr != sr:
        raise ValueError(f"Expected SR={sr}, got {file_sr}")
    if data.ndim == 2:
        # Use second channel (same convention as MaAI Wav input)
        data = data[:, 1]
    start = int(offset_sec * sr)
    end = start + int(crop_sec * sr)
    data = data[start:end]
    print(f"Loaded WAV: {wav_path}, samples={len(data)}, duration={len(data)/sr:.2f}s")
    return data


# ===========================================================================
# 2a. Load model — Approach A: evaluation_nod_para.py と同じ方法でロード
# ===========================================================================
def load_model_training(ckpt_path, device):
    """
    VAP_Nodding_para/vap/evaluation_nod_para.py と同じ方法:
      VAPModel.load_from_checkpoint(checkpoint, weights_only=False)
    """
    from vap.model_nod_para import VapConfig as VapConfigTrain
    from vap.train_nod_para import VAPModel, OptConfig, DataConfig

    # torch.load 時に __main__.OptConfig 等が必要なので登録
    import __main__
    _main_classes = {'OptConfig': OptConfig, 'DataConfig': DataConfig, 'VAPModel': VAPModel}
    _injected = []
    for name, cls in _main_classes.items():
        if not hasattr(__main__, name):
            setattr(__main__, name, cls)
            _injected.append(name)

    # checkpoint を peek して conf を取得 → maai_model_pt をローカルに差し替え
    torch.serialization.add_safe_globals([VapConfigTrain])
    ckpt_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt_peek.get("hyper_parameters", {})
    conf_obj = hp.get("conf", None)

    override_kwargs = {}
    if conf_obj is not None:
        orig_maai_pt = getattr(conf_obj, "maai_model_pt", "")
        if not os.path.exists(orig_maai_pt):
            _enc_name = os.path.basename(orig_maai_pt)
            for _search_dir in [_assets, _vap_root]:
                _local_enc = os.path.join(_search_dir, _enc_name)
                if os.path.exists(_local_enc):
                    conf_obj.maai_model_pt = _local_enc
                    override_kwargs["conf"] = conf_obj
                    print(f"  [maai_model_pt] {orig_maai_pt} -> {_local_enc}")
                    break

    # evaluation_nod_para.py と同じ: VAPModel.load_from_checkpoint
    model = VAPModel.load_from_checkpoint(
        ckpt_path, weights_only=False, **override_kwargs
    )
    model.eval()

    # nod_param_stats フォールバック
    if not hasattr(model, 'nod_param_stats') or model.nod_param_stats is None:
        model.nod_param_stats = {
            'range_mean': 0.0, 'range_std': 1.0,
            'speed_mean': 0.0, 'speed_std': 1.0,
            'swing_up_mean': 0.0, 'swing_up_std': 1.0,
        }

    train_frame_hz = model.conf.frame_hz
    print(f"[Training VAPModel] encoder_type={model.conf.encoder_type}, frame_hz={train_frame_hz}")
    print(f"  dim={model.conf.dim}, channel_layers={model.conf.channel_layers}, cross_layers={model.conf.cross_layers}")
    print(f"  nod_param_stats={model.nod_param_stats}")

    model.to(device)
    return model, train_frame_hz


# ===========================================================================
# 2b. Load model — Approach B/C: model.py の本番パスと同じルートでロード
# ===========================================================================
def load_model_maai(ckpt_path, frame_rate, device):
    """
    model.py の Maai.__init__(mode='nod_para') と同じロードパスを再現。
    チェックポイントから直接 MaAI の VapGPT_nod_para を構築する。
    """
    from maai.model import _setup_vap_compat, _cleanup_vap_compat, _build_nod_para_config
    from maai.models.vap_nod_para import VapGPT_nod_para

    print(f"[MaAI VapGPT_nod_para] Loading checkpoint from: {ckpt_path}")

    # 1. torch.load (model.py と同じ vap compat stubs を使用)
    _vap_stubs = _setup_vap_compat()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    finally:
        _cleanup_vap_compat(_vap_stubs)

    # 2. state_dict 抽出 (Lightning の 'model.' prefix を除去)
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
              for k, v in sd.items()}

    # 3. VapConfig 構築
    hp = ckpt.get("hyper_parameters", {})
    conf = _build_nod_para_config(hp)
    conf.frame_hz = frame_rate
    print(f"  encoder_type={conf.encoder_type}, inference_frame_hz={conf.frame_hz}")
    print(f"  dim={conf.dim}, channel_layers={conf.channel_layers}, cross_layers={conf.cross_layers}")

    # 4. n_heads 取得 (model.py と同じロジック)
    _conf_obj = hp.get("conf", None)
    _enc_n_heads = None
    if _conf_obj is not None and hasattr(_conf_obj, "encoder_n_heads"):
        _enc_n_heads = _conf_obj.encoder_n_heads
    if _enc_n_heads is None:
        _enc_n_heads = hp.get("encoder_n_heads", 8)
    print(f"  n_heads={_enc_n_heads}")

    # 5. モデル構築 & 重みロード
    model = VapGPT_nod_para(conf)
    model.load_encoder_from_state_dict(sd, frame_hz=frame_rate, lim_context_sec=-1, n_heads=_enc_n_heads)
    result = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={result.missing_keys[:3]}..., unexpected={result.unexpected_keys[:3]}...")

    # 6. nod_param_stats
    default_stats = {
        'range_mean': 0.0, 'range_std': 1.0,
        'speed_mean': 0.0, 'speed_std': 1.0,
        'swing_up_mean': 0.0, 'swing_up_std': 1.0,
    }
    nod_param_stats = ckpt.get("nod_param_stats", None)
    if nod_param_stats is None:
        nod_param_stats = hp.get("nod_param_stats", default_stats)
    model.nod_param_stats = nod_param_stats
    print(f"  nod_param_stats={nod_param_stats}")

    model.to(device)
    model.eval()
    return model


# ===========================================================================
# 3. Helper: 訓練コードの VapGPT.forward() でフル推論
# ===========================================================================
@torch.no_grad()
def run_training_forward(model_train, wav_ch1: np.ndarray, wav_ch2: np.ndarray, device):
    """
    訓練コードの VapGPT.forward(waveform) を呼ぶ。
    waveform: (B, 2, T) — ch0=ERICA(sp0), ch1=User(sp1)
    Returns: heads dict {p_nod, nod_count, nod_range, nod_speed, nod_swing_up}
    """
    # (B, 2, T): channel 0 = ERICA, channel 1 = User
    wav_erica_t = torch.from_numpy(wav_ch2).float().unsqueeze(0)  # (1, T)
    wav_user_t = torch.from_numpy(wav_ch1).float().unsqueeze(0)   # (1, T)
    waveform = torch.stack([wav_erica_t, wav_user_t], dim=1).to(device)  # (1, 2, T)

    out = model_train(waveform=waveform)

    # p_nod: (B, T, 1) -> sigmoid -> (T,)
    p_nod = out["nod"].sigmoid().squeeze(-1)[0].cpu()

    # nod_count: (B, T, C) -> argmax or sigmoid
    nod_count_raw = out["nod_count"]
    nod_count_binary = getattr(model_train.conf, 'nod_count_binary', 0) == 1
    if nod_count_binary:
        nod_count = nod_count_raw.sigmoid().squeeze(-1)[0].cpu()
    else:
        nod_count = nod_count_raw.argmax(dim=-1)[0].cpu().float()

    # nod_range: denormalize
    stats = model_train.nod_param_stats
    nod_range_z = out["nod_range"].squeeze(-1)[0].cpu()
    nod_range = nod_range_z * stats['range_std'] + stats['range_mean']

    # nod_speed: denormalize
    nod_speed_z = out["nod_speed"].squeeze(-1)[0].cpu()
    nod_speed = nod_speed_z * stats['speed_std'] + stats['speed_mean']

    # nod_swing_up_binary: sigmoid
    nod_swing_up = out["nod_swing_up_binary"].sigmoid().squeeze(-1)[0].cpu()

    heads = {
        'p_nod': p_nod,
        'nod_count': nod_count,
        'nod_range': nod_range,
        'nod_speed': nod_speed,
        'nod_swing_up': nod_swing_up,
    }
    return heads


# ===========================================================================
# 4. Helper: run encoder like MaAI (with padding + n_skip)
# ===========================================================================
@torch.no_grad()
def encode_like_maai_full(model, wav_ch1: np.ndarray, wav_ch2: np.ndarray, device):
    """
    Use MaAI's encode_audio with encoder_cache=None.
    When cache is None: no frame_contxt_padding, n_skip_frames=0 (matches training).
    """
    t1 = torch.from_numpy(wav_ch1).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    t2 = torch.from_numpy(wav_ch2).float().unsqueeze(0).unsqueeze(0).to(device)

    x1, x2, _ = model.encode_audio(t1, t2, encoder_cache=None)
    return x1, x2


# ===========================================================================
# 5. Helper: pseudo-streaming (chunk-by-chunk, no KV cache)
# ===========================================================================
@torch.no_grad()
def run_pseudo_streaming(model, wav_ch1: np.ndarray, wav_ch2: np.ndarray, device, frame_rate=10):
    """
    Simulate MaAI process() loop:
      - First chunk: buf=[], need 1600, encode with cache=None -> n_skip=0
      - Subsequent: buf=[overlap 640], need 2240, encode with cache -> n_skip=2
      - No GPT KV cache
    Returns: (e1_final, e2_final, heads_dict)
      heads_dict: {p_nod, nod_count, nod_range, nod_speed, nod_swing_up} — each is a list of per-frame values.
    """
    # CNN receptive field = 400 samples → need ≥400 overlap.
    # 640 = 2 × 320 (2 CNN frames) → n_skip=2
    frame_contxt_padding = 640
    encoder_n_skip = frame_contxt_padding // 320  # =2
    chunk_size = SAMPLE_RATE // frame_rate  # e.g., 1600 for 10Hz
    audio_frame_size = chunk_size + frame_contxt_padding

    # Buffers: first chunk empty, subsequent use overlap
    buf_x1 = np.array([], dtype=np.float32)
    buf_x2 = np.array([], dtype=np.float32)

    e1_list = []
    e2_list = []
    all_p_nod = []
    all_nod_count = []
    all_nod_range = []
    all_nod_speed = []
    all_nod_swing_up = []
    encoder_cache = None

    n_chunks = len(wav_ch1) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_x1 = wav_ch1[start:end]
        chunk_x2 = wav_ch2[start:end]

        buf_x1 = np.concatenate([buf_x1, chunk_x1])
        buf_x2 = np.concatenate([buf_x2, chunk_x2])

        min_size = chunk_size if encoder_cache is None else audio_frame_size
        if len(buf_x1) < min_size:
            continue

        t1 = torch.from_numpy(buf_x1).float().unsqueeze(0).unsqueeze(0).to(device)
        t2 = torch.from_numpy(buf_x2).float().unsqueeze(0).unsqueeze(0).to(device)
        n_skip = 0 if encoder_cache is None else encoder_n_skip
        e1, e2, encoder_cache = model.encode_audio(
            t1, t2, encoder_cache=encoder_cache, n_skip_frames=n_skip,
        )

        e1_list.append(e1)
        e2_list.append(e2)

        x1_full = torch.cat(e1_list, dim=1)
        x2_full = torch.cat(e2_list, dim=1)
        o1 = model.ar_channel(x1_full)
        o2 = model.ar_channel(x2_full)
        out = model.ar(o1["x"], o2["x"])
        x_out = out["x"]

        # p_nod (last frame)
        p_nod_frame = model.nod_head(x_out).sigmoid().squeeze(-1)[0, -1].cpu().item()
        all_p_nod.append(p_nod_frame)

        # Nod parameter heads (last frame)
        nod_param_input = x_out
        if model.use_nod_shared_encoder:
            nod_param_input = model.nod_shared_encoder(nod_param_input)
        nod_param_last = nod_param_input[:, -1:, :]  # (B, 1, D)

        # nod_count
        nod_count_raw = model.nod_count_head(nod_param_last)
        if model.conf.nod_count_binary == 1:
            all_nod_count.append(nod_count_raw.sigmoid().item())
        else:
            all_nod_count.append(float(nod_count_raw.argmax(dim=-1).item()))

        # nod_range (denormalized)
        nod_range_z = model.nod_range_head(nod_param_last).item()
        all_nod_range.append(
            nod_range_z * model.nod_param_stats['range_std'] + model.nod_param_stats['range_mean']
        )

        # nod_speed (denormalized)
        nod_speed_z = model.nod_speed_head(nod_param_last).item()
        all_nod_speed.append(
            nod_speed_z * model.nod_param_stats['speed_std'] + model.nod_param_stats['speed_mean']
        )

        # nod_swing_up_binary (sigmoid)
        all_nod_swing_up.append(
            model.nod_swing_up_binary_head(nod_param_last).sigmoid().item()
        )

        buf_x1 = buf_x1[-frame_contxt_padding:].copy()
        buf_x2 = buf_x2[-frame_contxt_padding:].copy()

    # Return final full encoder output for comparison
    if e1_list:
        e1_final = torch.cat(e1_list, dim=1)
        e2_final = torch.cat(e2_list, dim=1)
    else:
        e1_final = e2_final = None

    heads = {
        'p_nod': torch.tensor(all_p_nod),
        'nod_count': torch.tensor(all_nod_count),
        'nod_range': torch.tensor(all_nod_range),
        'nod_speed': torch.tensor(all_nod_speed),
        'nod_swing_up': torch.tensor(all_nod_swing_up),
    }
    return e1_final, e2_final, heads


# ===========================================================================
# 6. Approach D: Maai クラス経由 (本番 process() ループ)
# ===========================================================================
def run_maai_class(ckpt_path, wav_ch1: np.ndarray, wav_ch2: np.ndarray,
                   device, frame_rate=10, use_kv_cache=False):
    """
    Maai クラスを本番と同じように使い、process() にチャンクを直接投入して結果を収集。
    nod_para_mic.py と同じコードパスを通る。
    use_kv_cache=True で GPT の KV キャッシュを有効にする。
    """
    import queue as _queue
    from maai import Maai
    from maai.input import Base as _InputBase

    # ダミー入力（process() を直接呼ぶので実際の音声読み込みは不要）
    class _DummyInput(_InputBase):
        """subscribe() を持つだけのダミー入力"""
        def start(self): pass
        def stop(self): pass
        def get_audio_data(self, q): return None

    dummy1 = _DummyInput()
    dummy2 = _DummyInput()

    maai = Maai(
        mode="nod_para",
        lang="jp",
        frame_rate=frame_rate,
        context_len_sec=20,
        audio_ch1=dummy1,
        audio_ch2=dummy2,
        device=device,
        local_model=ckpt_path,
        use_kv_cache=use_kv_cache,
        print_process_time=False,
        use_anchor_frames=USE_ANCHOR_FRAMES,
    )

    # 1600 サンプル (=0.1s @ 10Hz) ずつ投入し F とフレーム境界を一致させる
    chunk_samples = 16000 // frame_rate  # 1600
    n_samples = min(len(wav_ch1), len(wav_ch2))

    all_p_nod = []
    all_nod_count = []
    all_nod_range = []
    all_nod_speed = []
    all_nod_swing_up = []

    for i in range(0, n_samples, chunk_samples):
        end = i + chunk_samples
        if end > n_samples:
            break
        chunk1 = wav_ch1[i:end]
        chunk2 = wav_ch2[i:end]
        maai.process(chunk1, chunk2)

        # process() が result_dict_queue に結果を入れたら回収
        while not maai.result_dict_queue.empty():
            try:
                r = maai.result_dict_queue.get_nowait()
            except _queue.Empty:
                break
            all_p_nod.append(float(r['p_nod']))
            # nod_count: softmax list の場合は argmax, scalar の場合はそのまま
            nc = r.get('nod_count', 0)
            if isinstance(nc, list):
                nc = float(max(range(len(nc)), key=lambda i: nc[i]))
            all_nod_count.append(float(nc))
            all_nod_range.append(float(r.get('nod_range', 0)))
            all_nod_speed.append(float(r.get('nod_speed', 0)))
            all_nod_swing_up.append(float(r.get('nod_swing_up_binary', 0)))

    heads = {
        'p_nod': torch.tensor(all_p_nod),
        'nod_count': torch.tensor(all_nod_count),
        'nod_range': torch.tensor(all_nod_range),
        'nod_speed': torch.tensor(all_nod_speed),
        'nod_swing_up': torch.tensor(all_nod_swing_up),
    }
    return heads


# ===========================================================================
# 7a. Approach F: Sliding-window full-sequence forward (training model)
# ===========================================================================
@torch.no_grad()
def run_sliding_window_forward(model_train, wav_ch1: np.ndarray, wav_ch2: np.ndarray,
                               device, frame_rate=10, context_sec=20.0):
    """
    学習コードの forward を 20 秒 sliding window で呼び出す。
    D と同じ時刻の最終フレームの全ヘッド出力を取得して比較用のテンソルを返す。

    毎フレーム（0.1 秒）ずつウィンドウを進め、ウィンドウ末尾フレームの値を収集。
    ウィンドウが context_sec 未満の場合は先頭からの部分列を使用。

    Returns:
        heads dict {p_nod, nod_count, nod_range, nod_speed, nod_swing_up}: each Tensor(T,)
    """
    sr = 16000
    context_samples = int(context_sec * sr)
    chunk_samples = sr // frame_rate  # 1600 (= 0.1s)
    n_total = min(len(wav_ch1), len(wav_ch2))
    n_frames = n_total // chunk_samples

    stats = model_train.nod_param_stats
    nod_count_binary = getattr(model_train.conf, 'nod_count_binary', 0) == 1

    all_p_nod = []
    all_nod_count = []
    all_nod_range = []
    all_nod_speed = []
    all_nod_swing_up = []

    print(f"  [Approach F] Processing {n_frames} frames with {context_sec}s sliding window...")
    for fi in range(n_frames):
        end = (fi + 1) * chunk_samples
        start = max(0, end - context_samples)
        w1 = wav_ch1[start:end]
        w2 = wav_ch2[start:end]

        wav_erica_t = torch.from_numpy(w2).float().unsqueeze(0)
        wav_user_t = torch.from_numpy(w1).float().unsqueeze(0)
        waveform = torch.stack([wav_erica_t, wav_user_t], dim=1).to(device)

        out = model_train(waveform=waveform)

        # 最終フレームの各ヘッド値を取得（D/E と同じ）
        p_last = out["nod"].sigmoid().squeeze(-1)[0, -1].cpu().item()
        all_p_nod.append(p_last)

        # nod_count (last frame)
        nod_count_raw = out["nod_count"]
        if nod_count_binary:
            all_nod_count.append(nod_count_raw.sigmoid().squeeze(-1)[0, -1].cpu().item())
        else:
            all_nod_count.append(float(nod_count_raw.argmax(dim=-1)[0, -1].cpu().item()))

        # nod_range (denormalized, last frame)
        nod_range_z = out["nod_range"].squeeze(-1)[0, -1].cpu().item()
        all_nod_range.append(nod_range_z * stats['range_std'] + stats['range_mean'])

        # nod_speed (denormalized, last frame)
        nod_speed_z = out["nod_speed"].squeeze(-1)[0, -1].cpu().item()
        all_nod_speed.append(nod_speed_z * stats['speed_std'] + stats['speed_mean'])

        # nod_swing_up_binary (sigmoid, last frame)
        all_nod_swing_up.append(
            out["nod_swing_up_binary"].sigmoid().squeeze(-1)[0, -1].cpu().item()
        )

        if (fi + 1) % 50 == 0 or fi == n_frames - 1:
            print(f"    frame {fi+1}/{n_frames}  p_nod={p_last:.4f}  window={start/sr:.1f}-{end/sr:.1f}s")

    return {
        'p_nod': torch.tensor(all_p_nod),
        'nod_count': torch.tensor(all_nod_count),
        'nod_range': torch.tensor(all_nod_range),
        'nod_speed': torch.tensor(all_nod_speed),
        'nod_swing_up': torch.tensor(all_nod_swing_up),
    }


# ===========================================================================
# 7b. Run GPT + ALL heads for full-sequence approaches (A and B)
# ===========================================================================
@torch.no_grad()
def run_gpt_heads(model, x1, x2):
    """Run GPT layers and ALL nod parameter heads, return per-frame predictions."""
    o1 = model.ar_channel(x1)
    o2 = model.ar_channel(x2)
    out = model.ar(o1["x"], o2["x"])
    x_out = out["x"]

    p_nod = model.nod_head(x_out).sigmoid().squeeze(-1)[0].cpu()

    # --- Nod parameter heads (全フレーム) ---
    nod_param_input = x_out
    if model.use_nod_shared_encoder:
        nod_param_input = model.nod_shared_encoder(nod_param_input)

    # nod_count
    nod_count_raw = model.nod_count_head(nod_param_input)  # (B, T, C)
    if model.conf.nod_count_binary == 1:
        nod_count = nod_count_raw.sigmoid().squeeze(-1)[0].cpu()
    else:
        nod_count = nod_count_raw.argmax(dim=-1)[0].cpu().float()

    # nod_range (denormalized)
    nod_range_z = model.nod_range_head(nod_param_input).squeeze(-1)[0].cpu()
    nod_range = nod_range_z * model.nod_param_stats['range_std'] + model.nod_param_stats['range_mean']

    # nod_speed (denormalized)
    nod_speed_z = model.nod_speed_head(nod_param_input).squeeze(-1)[0].cpu()
    nod_speed = nod_speed_z * model.nod_param_stats['speed_std'] + model.nod_param_stats['speed_mean']

    # nod_swing_up_binary (sigmoid)
    nod_swing_up = model.nod_swing_up_binary_head(nod_param_input).sigmoid().squeeze(-1)[0].cpu()

    heads = {
        'p_nod': p_nod,
        'nod_count': nod_count,
        'nod_range': nod_range,
        'nod_speed': nod_speed,
        'nod_swing_up': nod_swing_up,
    }
    return heads, x_out, o1["x"], o2["x"]


# ===========================================================================
# 7. Visualization helpers
# ===========================================================================
def _find_active_regions(arr):
    """1が連続する区間を [(start, end), ...] で返す (endは排他)"""
    regions = []
    in_region = False
    start = 0
    for i in range(len(arr)):
        if arr[i] > 0 and not in_region:
            start = i
            in_region = True
        elif arr[i] <= 0 and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(arr)))
    return regions


# Approach colors / styles
_STYLES = {
    'A': dict(color='#1f77b4', linewidth=1.2, alpha=0.9, label='A: Training full-seq'),
    'B': dict(color='#2ca02c', linewidth=1.2, alpha=0.9, linestyle='--', label='B: MaAI full'),
    'C': dict(color='#d62728', linewidth=1.0, alpha=0.8, label='C: Streaming'),
    'D': dict(color='#9467bd', linewidth=1.0, alpha=0.8, linestyle=':', label='D: Maai (no KV)'),
    'E': dict(color='#ff7f0e', linewidth=1.0, alpha=0.8, linestyle='-.', label='E: Maai (KV cache)'),
    'F': dict(color='#17becf', linewidth=1.2, alpha=0.9, linestyle='--', label='F: Training 20s SW'),
}


def visualize_three_approaches(
    wav_user: np.ndarray,
    wav_erica: np.ndarray,
    heads_A: dict | None,
    heads_B: dict | None,
    heads_C: dict | None,
    frame_hz: float = 10,
    sample_rate: int = 16000,
    save_dir: str = "visualizations",
    nod_threshold: float = 0.5,
    heads_D: dict = None,
    heads_E: dict = None,
    heads_F: dict = None,
):
    """
    全アプローチの全ヘッド出力を可視化して保存する。

    レイアウト (6行):
      Row 1: Waveform (User + ERICA)
      Row 2–6: 各ヘッド — A, B, C, D, E, F 重畳
    """
    os.makedirs(save_dir, exist_ok=True)

    approaches = [(n, h) for n, h in [('A', heads_A), ('B', heads_B), ('C', heads_C)] if h is not None]
    if heads_D is not None:
        approaches.append(('D', heads_D))
    if heads_E is not None:
        approaches.append(('E', heads_E))
    if heads_F is not None:
        approaches.append(('F', heads_F))

    _ref_heads = heads_D if heads_D is not None else (heads_F if heads_F is not None else heads_E)
    n_frames = len(_ref_heads['p_nod']) if _ref_heads is not None else 0
    time_axis = np.arange(n_frames) / frame_hz
    max_time = time_axis[-1]

    # --- Waveform downsampling for display ---
    max_display_samples = 50000
    wav_user_disp = wav_user
    wav_erica_disp = wav_erica
    disp_sr = sample_rate
    if len(wav_user) > max_display_samples:
        step = len(wav_user) // max_display_samples
        wav_user_disp = wav_user[::step]
        wav_erica_disp = wav_erica[::step]
        disp_sr = sample_rate / step
    time_axis_wav = np.arange(len(wav_user_disp)) / disp_sr

    # Convert to numpy
    def _np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    # ===================================================================
    # Figure 1: 6-row combined plot
    # ===================================================================
    fig = plt.figure(figsize=(22, 18))
    n_rows = 6
    gs = fig.add_gridspec(n_rows, 1, hspace=0.30, top=0.95, bottom=0.04)

    row = 0
    # ----- Row 1: Waveform -----
    ax = fig.add_subplot(gs[row, 0])
    ax.plot(time_axis_wav, wav_user_disp, color='red', linewidth=0.4, alpha=0.7, label='User (sp1)')
    ax.plot(time_axis_wav, wav_erica_disp, color='blue', linewidth=0.4, alpha=0.5, label='ERICA (sp0)')
    ax.set_ylabel('Waveform', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    row += 1

    # ----- Row 2: p_nod (NOD Timing) -----
    ax = fig.add_subplot(gs[row, 0])
    for key, heads in approaches:
        vals = _np(heads['p_nod'])
        t = np.arange(len(vals)) / frame_hz
        ax.plot(t, vals, **_STYLES[key])
    # Threshold line
    ax.axhline(y=nod_threshold, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    # 閾値超え区間を薄く塗りつぶし（A があれば A、なければ D を基準）
    _ref_for_regions = heads_A if heads_A is not None else heads_D
    if _ref_for_regions is not None and 'p_nod' in _ref_for_regions:
        pnod_ref = _np(_ref_for_regions['p_nod'])
        pred_binary = (pnod_ref >= nod_threshold).astype(float)
        regions = _find_active_regions(pred_binary)
        _label = 'A >= threshold' if heads_A is not None else 'D >= threshold'
        for idx_r, (rs, re) in enumerate(regions):
            t_start = rs / frame_hz
            t_end = (re - 1) / frame_hz
            ax.axvspan(t_start, t_end, alpha=0.10, color='#1f77b4',
                       label=_label if idx_r == 0 else '')
    ax.set_ylabel('p_nod', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    row += 1

    # ----- Row 3: nod_count -----
    ax = fig.add_subplot(gs[row, 0])
    for key, heads in approaches:
        if 'nod_count' not in heads:
            continue
        vals = _np(heads['nod_count'])
        t = np.arange(len(vals)) / frame_hz
        ax.step(t, vals, where='mid', **_STYLES[key])
    ax.set_ylabel('nod_count', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.set_ylim([-0.3, 2.5])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['1x', '2x', '3x+'])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    row += 1

    # ----- Row 4: nod_range -----
    ax = fig.add_subplot(gs[row, 0])
    for key, heads in approaches:
        if 'nod_range' not in heads:
            continue
        vals = _np(heads['nod_range'])
        t = np.arange(len(vals)) / frame_hz
        ax.plot(t, vals, **_STYLES[key])
    ax.set_ylabel('nod_range', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    row += 1

    # ----- Row 5: nod_speed -----
    ax = fig.add_subplot(gs[row, 0])
    for key, heads in approaches:
        if 'nod_speed' not in heads:
            continue
        vals = _np(heads['nod_speed'])
        t = np.arange(len(vals)) / frame_hz
        ax.plot(t, vals, **_STYLES[key])
    ax.set_ylabel('nod_speed', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    row += 1

    # ----- Row 6: nod_swing_up_binary -----
    ax = fig.add_subplot(gs[row, 0])
    for key, heads in approaches:
        if 'nod_swing_up' not in heads:
            continue
        vals = _np(heads['nod_swing_up'])
        t = np.arange(len(vals)) / frame_hz
        ax.plot(t, vals, **_STYLES[key])
    ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('swing_up', fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_xlim([0, max_time])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Statistics in suptitle ---
    title_parts = []
    if heads_A is not None and heads_C is not None:
        pnod_a_np = _np(heads_A['p_nod'])
        pnod_c_np = _np(heads_C['p_nod'])
        n_a = int((pnod_a_np >= nod_threshold).sum())
        n_c = int((pnod_c_np >= nod_threshold).sum())
        min_n = min(len(pnod_a_np), len(pnod_c_np))
        diff_ac = float(np.abs(pnod_a_np[:min_n] - pnod_c_np[:min_n]).max())
        title_parts.extend([
            f"A vs C p_nod max_diff={diff_ac:.4f}",
            f"A>={nod_threshold}: {n_a}fr, C>={nod_threshold}: {n_c}fr",
        ])
        if heads_D is not None:
            pnod_d_np = _np(heads_D['p_nod'])
            n_d = int((pnod_d_np >= nod_threshold).sum())
            min_n_d = min(len(pnod_a_np), len(pnod_d_np))
            diff_ad = float(np.abs(pnod_a_np[:min_n_d] - pnod_d_np[:min_n_d]).max())
            title_parts.append(f"A vs D={diff_ad:.4f}")
        if heads_E is not None:
            pnod_e_np = _np(heads_E['p_nod'])
            n_e = int((pnod_e_np >= nod_threshold).sum())
            min_n_e = min(len(pnod_a_np), len(pnod_e_np))
            diff_ae = float(np.abs(pnod_a_np[:min_n_e] - pnod_e_np[:min_n_e]).max())
            title_parts.append(f"A vs E(KV)={diff_ae:.4f}")
        if heads_F is not None:
            pnod_f_np = _np(heads_F['p_nod'])
            min_n_f = min(len(pnod_a_np), len(pnod_f_np))
            diff_af = float(np.abs(pnod_a_np[:min_n_f] - pnod_f_np[:min_n_f]).max())
            title_parts.append(f"A vs F(SW)={diff_af:.4f}")
    elif heads_D is not None:
        pnod_d_np = _np(heads_D['p_nod'])
        n_d = int((pnod_d_np >= nod_threshold).sum())
        title_parts.append(f"D>={nod_threshold}: {n_d}fr")
        if heads_E is not None:
            pnod_e_np = _np(heads_E['p_nod'])
            min_n_de = min(len(pnod_d_np), len(pnod_e_np))
            diff_de = float(np.abs(pnod_d_np[:min_n_de] - pnod_e_np[:min_n_de]).max())
            title_parts.append(f"D vs E={diff_de:.4f}")
        if heads_F is not None:
            pnod_f_np = _np(heads_F['p_nod'])
            min_n_df = min(len(pnod_d_np), len(pnod_f_np))
            diff_df = float(np.abs(pnod_d_np[:min_n_df] - pnod_f_np[:min_n_df]).max())
            title_parts.append(f"D vs F={diff_df:.4f}")
    title = "Compare Approaches  |  " + "  |  ".join(title_parts) if title_parts else "Compare Approaches (D/E/F)"
    plt.suptitle(title, fontsize=9, fontweight='bold', y=0.98)

    save_path = os.path.join(save_dir, "all_heads_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close(fig)

    # ===================================================================
    # Figure 2–6: Individual head plots (larger, with diff subplot)
    # ===================================================================
    head_configs = [
        ('p_nod', 'p_nod (NOD Timing)', (-0.05, 1.05), 'plot'),
        ('nod_count', 'nod_count', (-0.3, 2.5), 'step'),
        ('nod_range', 'nod_range (denormalized)', None, 'plot'),
        ('nod_speed', 'nod_speed (denormalized)', None, 'plot'),
        ('nod_swing_up', 'nod_swing_up_binary (sigmoid)', (-0.05, 1.05), 'plot'),
    ]

    for head_key, head_title, ylim, plot_type in head_configs:
        fig, axes = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.25})

        # Top: overlaid A, B, C, D, E, F
        ax = axes[0]
        for key, heads in approaches:
            if head_key not in heads:
                continue
            vals = _np(heads[head_key])
            t = np.arange(len(vals)) / frame_hz
            if plot_type == 'step':
                ax.step(t, vals, where='mid', **_STYLES[key])
            else:
                ax.plot(t, vals, **_STYLES[key])

        if head_key == 'p_nod':
            ax.axhline(y=nod_threshold, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        if head_key == 'nod_swing_up':
            ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        if head_key == 'nod_count':
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['1x', '2x', '3x+'])

        ax.set_ylabel(head_title, fontsize=12)
        ax.set_xlim([0, max_time])
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Bottom: diff (A vs C/D/E/F、または D vs E/F)
        ax2 = axes[1]
        vals_a = _np(heads_A[head_key]) if heads_A is not None and head_key in heads_A else None
        vals_ref = vals_a if vals_a is not None else (_np(heads_D[head_key]) if heads_D is not None and head_key in heads_D else None)
        suptitle_parts = []

        if vals_a is not None and heads_C is not None and head_key in heads_C:
            vals_c = _np(heads_C[head_key])
            min_n = min(len(vals_a), len(vals_c))
            diff_ac = vals_a[:min_n] - vals_c[:min_n]
            t_diff = np.arange(min_n) / frame_hz
            ax2.fill_between(t_diff, diff_ac, 0, alpha=0.3, color='#d62728')
            ax2.plot(t_diff, diff_ac, color='#d62728', linewidth=0.8, alpha=0.7, label='A - C')
            suptitle_parts.append(f"A vs C: max={float(np.abs(diff_ac).max()):.6f}")

        if vals_a is not None and heads_D is not None and head_key in heads_D:
            vals_d = _np(heads_D[head_key])
            min_n_d = min(len(vals_a), len(vals_d))
            diff_ad = vals_a[:min_n_d] - vals_d[:min_n_d]
            t_diff_d = np.arange(min_n_d) / frame_hz
            ax2.plot(t_diff_d, diff_ad, color='#9467bd', linewidth=0.8, alpha=0.7, linestyle=':', label='A - D')
            suptitle_parts.append(f"A vs D: max={float(np.abs(diff_ad).max()):.6f}")
        if vals_a is not None and heads_E is not None and head_key in heads_E:
            vals_e = _np(heads_E[head_key])
            min_n_e = min(len(vals_a), len(vals_e))
            diff_ae = vals_a[:min_n_e] - vals_e[:min_n_e]
            t_diff_e = np.arange(min_n_e) / frame_hz
            ax2.plot(t_diff_e, diff_ae, color='#ff7f0e', linewidth=0.8, alpha=0.7, linestyle='-.', label='A - E(KV)')
            suptitle_parts.append(f"A vs E(KV): max={float(np.abs(diff_ae).max()):.6f}")
        if vals_a is not None and heads_F is not None and head_key in heads_F:
            vals_f = _np(heads_F[head_key])
            min_n_f = min(len(vals_a), len(vals_f))
            diff_af = vals_a[:min_n_f] - vals_f[:min_n_f]
            t_diff_f = np.arange(min_n_f) / frame_hz
            ax2.plot(t_diff_f, diff_af, color='#17becf', linewidth=0.8, alpha=0.7, linestyle='--', label='A - F(SW)')
            suptitle_parts.append(f"A vs F(SW): max={float(np.abs(diff_af).max()):.6f}")

        # A が無い場合: D vs E, D vs F
        if vals_ref is None and heads_D is not None and head_key in heads_D:
            vals_d = _np(heads_D[head_key])
            vals_ref = vals_d
        if vals_ref is not None and heads_E is not None and head_key in heads_E and vals_a is None:
            vals_e = _np(heads_E[head_key])
            min_n = min(len(vals_ref), len(vals_e))
            diff = vals_ref[:min_n] - vals_e[:min_n]
            t_diff = np.arange(min_n) / frame_hz
            ax2.plot(t_diff, diff, color='#ff7f0e', linewidth=0.8, alpha=0.7, linestyle='-.', label='D - E(KV)')
            suptitle_parts.append(f"D vs E: max={float(np.abs(diff).max()):.6f}")
        if vals_ref is not None and heads_F is not None and head_key in heads_F and vals_a is None:
            vals_f = _np(heads_F[head_key])
            min_n = min(len(vals_ref), len(vals_f))
            diff = vals_ref[:min_n] - vals_f[:min_n]
            t_diff = np.arange(min_n) / frame_hz
            ax2.plot(t_diff, diff, color='#17becf', linewidth=0.8, alpha=0.7, linestyle='--', label='D - F(SW)')
            suptitle_parts.append(f"D vs F: max={float(np.abs(diff).max()):.6f}")

        ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax2.set_ylabel('Diff', fontsize=11)
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_xlim([0, max_time])
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{head_title}  |  {', '.join(suptitle_parts)}",
                     fontsize=10, fontweight='bold')

        fname = f"{head_key}_timeseries.png"
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)

    print(f"\n  All visualizations saved to: {save_dir}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 80)
    print("Comparison of Six Approaches (A/B/C/D/E/F)")
    print("=" * 80)

    # Load WAV
    wav_user = load_crop_wav(WAV_FILE, CROP_OFFSET_SEC, CROP_SEC)
    # ERICA channel: white noise (same as test setup)
    rng = np.random.RandomState(42)
    wav_erica = rng.randn(len(wav_user)).astype(np.float32) * 1e-4

    audio_duration_sec = len(wav_user) / SAMPLE_RATE
    rtf_results = {}  # {approach_name: (elapsed_sec, rtf)}

    # ===================================================================
    # Load models
    # ===================================================================
    # Approach A & F: 訓練コードの VapGPT を使用
    print("\n--- Loading model for Approach A/F (training VapGPT) ---")
    model_train, train_frame_hz = load_model_training(CHECKPOINT, DEVICE)

    # Approach B/C: model.py と同じ本番ロードパス
    print("\n--- Loading model for Approach B/C (MaAI VapGPT_nod_para) ---")
    model_maai = load_model_maai(CHECKPOINT, FRAME_RATE, DEVICE)

    heads_A, heads_B, heads_C = None, None, None
    p_nod_A, p_nod_B, p_nod_C = None, None, None
    e2_C, x2_B = None, None

    if RUN_ABC:
        # ===================================================================
        # Approach A: 訓練コードの VapGPT.forward() でフル推論
        # ===================================================================
        print("\n--- Approach A: Training VapGPT.forward() ---")
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        heads_A = run_training_forward(model_train, wav_user, wav_erica, DEVICE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_A = time.perf_counter() - t0
        rtf_results['A'] = (elapsed_A, elapsed_A / audio_duration_sec)
        p_nod_A = heads_A['p_nod']
        print(f"  p_nod frames: {len(p_nod_A)}")
        print(f"  p_nod: min={p_nod_A.min():.4f}, max={p_nod_A.max():.4f}, mean={p_nod_A.mean():.4f}")
        print(f"  Frames > 0.5: {(p_nod_A > 0.5).sum().item()}")
        print(f"  Time: {elapsed_A:.3f}s, RTF: {rtf_results['A'][1]:.4f}")

        # ===================================================================
        # Approach B: MaAI encode_audio (cache=None -> no padding, n_skip=0)
        # ===================================================================
        print("\n--- Approach B: MaAI full-sequence (no padding, n_skip=0) ---")
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x1_B, x2_B = encode_like_maai_full(model_maai, wav_user, wav_erica, DEVICE)
        heads_B, x_out_B, o1_B, o2_B = run_gpt_heads(model_maai, x1_B, x2_B)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_B = time.perf_counter() - t0
        rtf_results['B'] = (elapsed_B, elapsed_B / audio_duration_sec)
        p_nod_B = heads_B['p_nod']
        print(f"  Encoder output: x1={list(x1_B.shape)}, x2={list(x2_B.shape)}")
        print(f"  p_nod: min={p_nod_B.min():.4f}, max={p_nod_B.max():.4f}, mean={p_nod_B.mean():.4f}")
        print(f"  Frames > 0.5: {(p_nod_B > 0.5).sum().item()}")
        print(f"  Time: {elapsed_B:.3f}s, RTF: {rtf_results['B'][1]:.4f}")

        # ===================================================================
        # Approach C: Pseudo-streaming (chunk by chunk, no KV cache)
        # ===================================================================
        print("\n--- Approach C: MaAI pseudo-streaming (no KV cache) ---")
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        e1_C, e2_C, heads_C = run_pseudo_streaming(model_maai, wav_user, wav_erica, DEVICE, FRAME_RATE)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_C = time.perf_counter() - t0
        rtf_results['C'] = (elapsed_C, elapsed_C / audio_duration_sec)
        p_nod_C = heads_C['p_nod']
        print(f"  Encoder output: e1={list(e1_C.shape) if e1_C is not None else None}")
        print(f"  p_nod frames: {len(p_nod_C)}")
        print(f"  p_nod: min={p_nod_C.min():.4f}, max={p_nod_C.max():.4f}, mean={p_nod_C.mean():.4f}")
        print(f"  Frames > 0.5: {(p_nod_C > 0.5).sum().item()}")
        print(f"  Time: {elapsed_C:.3f}s, RTF: {rtf_results['C'][1]:.4f}")
    else:
        print("\n--- Skipping Approach A, B, C (RUN_ABC=False) ---")

    # ===================================================================
    # Approach D: Maai クラス経由 (本番 process() ループ)
    # ===================================================================
    print("\n--- Approach D: Maai class (production process() loop, no KV cache) ---")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    heads_D = run_maai_class(CHECKPOINT, wav_user, wav_erica, DEVICE, FRAME_RATE,
                             use_kv_cache=False)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed_D = time.perf_counter() - t0
    rtf_results['D'] = (elapsed_D, elapsed_D / audio_duration_sec)
    p_nod_D = heads_D['p_nod']
    print(f"  p_nod frames: {len(p_nod_D)}")
    print(f"  p_nod: min={p_nod_D.min():.4f}, max={p_nod_D.max():.4f}, mean={p_nod_D.mean():.4f}")
    print(f"  Frames > 0.5: {(p_nod_D > 0.5).sum().item()}")
    print(f"  Time: {elapsed_D:.3f}s, RTF: {rtf_results['D'][1]:.4f}")

    # ===================================================================
    # Approach E: Maai クラス + KV キャッシュあり
    # ===================================================================
    print("\n--- Approach E: Maai class (production process() loop, WITH KV cache) ---")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    heads_E = run_maai_class(CHECKPOINT, wav_user, wav_erica, DEVICE, FRAME_RATE,
                             use_kv_cache=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed_E = time.perf_counter() - t0
    rtf_results['E'] = (elapsed_E, elapsed_E / audio_duration_sec)
    p_nod_E = heads_E['p_nod']
    print(f"  p_nod frames: {len(p_nod_E)}")
    print(f"  p_nod: min={p_nod_E.min():.4f}, max={p_nod_E.max():.4f}, mean={p_nod_E.mean():.4f}")
    print(f"  Frames > 0.5: {(p_nod_E > 0.5).sum().item()}")
    print(f"  Time: {elapsed_E:.3f}s, RTF: {rtf_results['E'][1]:.4f}")

    # ===================================================================
    # Approach F: 学習コードの forward を 20 秒 sliding window で実行
    # ===================================================================
    print("\n--- Approach F: Training forward with 20s sliding window ---")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    heads_F = run_sliding_window_forward(model_train, wav_user, wav_erica, DEVICE,
                                         FRAME_RATE, context_sec=20.0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed_F = time.perf_counter() - t0
    rtf_results['F'] = (elapsed_F, elapsed_F / audio_duration_sec)
    p_nod_F = heads_F['p_nod']
    print(f"  p_nod frames: {len(p_nod_F)}")
    print(f"  p_nod: min={p_nod_F.min():.4f}, max={p_nod_F.max():.4f}, mean={p_nod_F.mean():.4f}")
    print(f"  Frames > 0.5: {(p_nod_F > 0.5).sum().item()}")
    print(f"  Time: {elapsed_F:.3f}s, RTF: {rtf_results['F'][1]:.4f}")

    # ===================================================================
    # D vs F comparison (same sliding window, different pipeline)
    # ===================================================================
    min_len_df = min(len(p_nod_D), len(p_nod_F))
    diff_df = (p_nod_D[:min_len_df] - p_nod_F[:min_len_df]).abs()
    print(f"\n--- D vs F (sliding window: training forward vs Maai process) ---")
    print(f"  max diff = {diff_df.max():.6f}, mean diff = {diff_df.mean():.6f}")
    # 20 秒以降の差分
    start_200 = min(200, min_len_df)
    if min_len_df > start_200:
        diff_df_after20 = diff_df[start_200:]
        print(f"  after 20s: max diff = {diff_df_after20.max():.6f}, mean diff = {diff_df_after20.mean():.6f}")

    # 診断: 20秒以降の特定フレームでの D vs F 値
    print("\n--- Diagnostic: D vs F at frame 199, 249, 299 (20s以降) ---")
    chunk_samples = SAMPLE_RATE // FRAME_RATE
    for fi in [199, 249, 299]:
        if fi < min_len_df:
            t_sec = (fi + 1) * chunk_samples / SAMPLE_RATE
            d_val = p_nod_D[fi].item()
            f_val = p_nod_F[fi].item()
            print(f"  frame {fi} (t={t_sec:.1f}s): D={d_val:.4f}, F={f_val:.4f}, |D-F|={abs(d_val-f_val):.4f}")

    # B vs C エンコーダ段階別比較
    if RUN_ABC:
        print("\n--- B vs C encoder stage comparison (User channel) ---")
        with torch.no_grad():
            enc = model_maai.encoder
            chunk_size = SAMPLE_RATE // FRAME_RATE  # 1600
            fcp = 640  # frame_contxt_padding (2 CNN frames for full receptive field coverage)
            enc_n_skip = fcp // 320  # =2
            n_chunks = len(wav_user) // chunk_size

            # --- B: フルシーケンスの各段階 ---
            wav_t = torch.from_numpy(wav_user).float().unsqueeze(0).to(DEVICE)
            lengths_full = torch.tensor([wav_t.shape[1]], device=DEVICE)
            feats_full, fl_full = enc.model.frontend(wav_t, lengths_full)  # 50Hz CNN出力
            proj_full = enc.model.proj(feats_full)
            x_full = proj_full
            for blk in enc.model.blocks_50:
                x_full, _, _ = blk(x_full, key_padding_mask=None, past_k=None, past_v=None)
            norm_full = enc.model.norm(x_full)  # (1, 1000, 512) 50Hz
            ds_full = enc.downsample(norm_full)  # (1, 200, 512) 10Hz

            # --- C: チャンク別の各段階を手動実行 ---
            buf = np.array([], dtype=np.float32)
            cnn_frames_list = []  # 50Hz CNN出力フレームを蓄積
            transformer_cache = None  # {blocks_50: [(k,v), ...]}
            norm_frames_list = []  # 50Hz norm出力
            ds_frames_list = []    # 10Hz downsample出力
            ds_buf = None          # downsample cache (last kernel-1 frames)
            chunk_idx = 0

            for i in range(n_chunks):
                start = i * chunk_size
                end = start + chunk_size
                buf = np.concatenate([buf, wav_user[start:end]])
                min_sz = chunk_size if transformer_cache is None else (chunk_size + fcp)
                if len(buf) < min_sz:
                    continue

                wav_chunk = torch.from_numpy(buf).float().unsqueeze(0).to(DEVICE)
                lengths_chunk = torch.tensor([wav_chunk.shape[1]], device=DEVICE)

                # CNN frontend
                feats_c, fl_c = enc.model.frontend(wav_chunk, lengths_chunk)
                proj_c = enc.model.proj(feats_c)

                # n_skip
                n_skip = 0 if transformer_cache is None else enc_n_skip
                x_c = proj_c[:, n_skip:, :] if n_skip > 0 else proj_c

                # Transformer with KV cache
                if transformer_cache is not None:
                    blk_cache = transformer_cache.get("blocks_50", [None] * len(enc.model.blocks_50))
                else:
                    blk_cache = [None] * len(enc.model.blocks_50)
                new_cache_list = []
                for bi, blk in enumerate(enc.model.blocks_50):
                    past_kv = blk_cache[bi]
                    pk, pv = past_kv if past_kv is not None else (None, None)
                    x_c, new_k, new_v = blk(x_c, key_padding_mask=None, past_k=pk, past_v=pv)
                    new_cache_list.append((new_k, new_v))
                transformer_cache = {"blocks_50": new_cache_list}

                # Norm
                norm_c = enc.model.norm(x_c)
                norm_frames_list.append(norm_c)

                # Downsample (per-chunk with cache)
                if enc.downsample is not None:
                    kernel_size = enc.downsample[1].kernel_size[0]  # 5
                    new_ds_buf = norm_c[:, -(kernel_size - 1):, :].clone()
                    if ds_buf is not None:
                        z_ctx = torch.cat([ds_buf, norm_c], dim=1)
                        x_ds = z_ctx.transpose(1, 2)
                        cconv = enc.downsample[1]
                        x_ds = F.conv1d(x_ds, cconv.weight, cconv.bias,
                                        stride=cconv.stride, dilation=cconv.dilation)
                        x_ds = enc.downsample[2](x_ds)
                        x_ds = enc.downsample[3](x_ds)
                        ds_c = x_ds.transpose(1, 2)
                    else:
                        ds_c = enc.downsample(norm_c)
                    ds_buf = new_ds_buf
                else:
                    ds_c = norm_c
                ds_frames_list.append(ds_c)

                # CNN frames (after skip)
                cnn_c = proj_c[:, n_skip:, :]
                cnn_frames_list.append(cnn_c)

                buf = buf[-fcp:].copy()
                chunk_idx += 1

            # 蓄積結果を結合
            cnn_cat = torch.cat(cnn_frames_list, dim=1)   # (1, ~1000, 512)
            norm_cat = torch.cat(norm_frames_list, dim=1)  # (1, ~1000, 512)
            ds_cat = torch.cat(ds_frames_list, dim=1)      # (1, ~200, 512)

            # 50Hz CNN 出力比較
            min_l = min(proj_full.shape[1], cnn_cat.shape[1])
            cnn_diff = (proj_full[:, :min_l, :] - cnn_cat[:, :min_l, :]).abs()
            print(f"  50Hz CNN+proj: B={list(proj_full.shape)}, C={list(cnn_cat.shape)}")
            print(f"    max_diff={cnn_diff.max():.6f}, mean_diff={cnn_diff.mean():.6f}")
            # Per-frame max diff (first 10 chunks = 50 frames)
            per_frame_cnn = cnn_diff[0].max(dim=-1).values[:50]
            print(f"    per-frame max (first 50): {[f'{v:.4f}' for v in per_frame_cnn.tolist()]}")

            # 50Hz transformer+norm 出力比較
            min_l2 = min(norm_full.shape[1], norm_cat.shape[1])
            norm_diff = (norm_full[:, :min_l2, :] - norm_cat[:, :min_l2, :]).abs()
            print(f"  50Hz norm: B={list(norm_full.shape)}, C={list(norm_cat.shape)}")
            print(f"    max_diff={norm_diff.max():.6f}, mean_diff={norm_diff.mean():.6f}")
            per_frame_norm = norm_diff[0].max(dim=-1).values[:50]
            print(f"    per-frame max (first 50): {[f'{v:.4f}' for v in per_frame_norm.tolist()]}")

            # 10Hz downsample 出力比較
            min_l3 = min(ds_full.shape[1], ds_cat.shape[1])
            ds_diff = (ds_full[:, :min_l3, :] - ds_cat[:, :min_l3, :]).abs()
            print(f"  10Hz downsample: B={list(ds_full.shape)}, C={list(ds_cat.shape)}")
            print(f"    max_diff={ds_diff.max():.6f}, mean_diff={ds_diff.mean():.6f}")
            per_frame_ds = ds_diff[0].max(dim=-1).values[:20]
            print(f"    per-frame max (first 20): {[f'{v:.4f}' for v in per_frame_ds.tolist()]}")

    # ===================================================================
    # RTF Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print(f"RTF Summary  (audio duration = {audio_duration_sec:.1f}s, device = {DEVICE})")
    print("=" * 80)
    print(f"  {'Approach':<40s}  {'Time (s)':>10s}  {'RTF':>10s}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*10}")
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key not in rtf_results:
            continue
        elapsed, rtf = rtf_results[key]
        desc = _STYLES[key]['label']
        print(f"  {desc:<40s}  {elapsed:>10.3f}  {rtf:>10.4f}")
    print()

    # ===================================================================
    # Comparison (console summary)
    # ===================================================================
    print("=" * 80)
    print("Comparison Summary")
    print("=" * 80)

    # Encoder比較: B vs C
    if e2_C is not None and x2_B is not None:
        min_len_BC = min(x2_B.shape[1], e2_C.shape[1])
        enc_diff_BC = (x2_B[:, :min_len_BC, :] - e2_C[:, :min_len_BC, :]).abs()
        print(f"\n  Encoder x2 diff (B vs C): max={enc_diff_BC.max():.6f}, mean={enc_diff_BC.mean():.6f}")

    # p_nod 比較
    if RUN_ABC:
        min_pnod_AB = min(len(p_nod_A), len(p_nod_B))
        pnod_diff_AB = (p_nod_A[:min_pnod_AB] - p_nod_B[:min_pnod_AB]).abs()
        print(f"\n  p_nod diff (A vs B): max={pnod_diff_AB.max():.6f}, mean={pnod_diff_AB.mean():.6f}")

        min_pnod_AC = min(len(p_nod_A), len(p_nod_C))
        pnod_diff_AC = (p_nod_A[:min_pnod_AC] - p_nod_C[:min_pnod_AC]).abs()
        print(f"  p_nod diff (A vs C): max={pnod_diff_AC.max():.6f}, mean={pnod_diff_AC.mean():.6f}")

        min_pnod_BC = min(len(p_nod_B), len(p_nod_C))
        pnod_diff_BC = (p_nod_B[:min_pnod_BC] - p_nod_C[:min_pnod_BC]).abs()
        print(f"  p_nod diff (B vs C): max={pnod_diff_BC.max():.6f}, mean={pnod_diff_BC.mean():.6f}")

        min_pnod_AD = min(len(p_nod_A), len(p_nod_D))
        pnod_diff_AD = (p_nod_A[:min_pnod_AD] - p_nod_D[:min_pnod_AD]).abs()
        print(f"\n  p_nod diff (A vs D): max={pnod_diff_AD.max():.6f}, mean={pnod_diff_AD.mean():.6f}")

        min_pnod_CD = min(len(p_nod_C), len(p_nod_D))
        pnod_diff_CD = (p_nod_C[:min_pnod_CD] - p_nod_D[:min_pnod_CD]).abs()
        print(f"  p_nod diff (C vs D): max={pnod_diff_CD.max():.6f}, mean={pnod_diff_CD.mean():.6f}")

        min_pnod_AE = min(len(p_nod_A), len(p_nod_E))
        pnod_diff_AE = (p_nod_A[:min_pnod_AE] - p_nod_E[:min_pnod_AE]).abs()
        print(f"\n  p_nod diff (A vs E): max={pnod_diff_AE.max():.6f}, mean={pnod_diff_AE.mean():.6f}")

        min_pnod_AF = min(len(p_nod_A), len(p_nod_F))
        pnod_diff_AF = (p_nod_A[:min_pnod_AF] - p_nod_F[:min_pnod_AF]).abs()
        print(f"\n  p_nod diff (A vs F): max={pnod_diff_AF.max():.6f}, mean={pnod_diff_AF.mean():.6f}")

    min_pnod_DE = min(len(p_nod_D), len(p_nod_E))
    pnod_diff_DE = (p_nod_D[:min_pnod_DE] - p_nod_E[:min_pnod_DE]).abs()
    print(f"  p_nod diff (D vs E): max={pnod_diff_DE.max():.6f}, mean={pnod_diff_DE.mean():.6f}")

    # F との比較 (学習コードの 20s sliding window)
    min_pnod_DF = min(len(p_nod_D), len(p_nod_F))
    pnod_diff_DF = (p_nod_D[:min_pnod_DF] - p_nod_F[:min_pnod_DF]).abs()
    print(f"  p_nod diff (D vs F): max={pnod_diff_DF.max():.6f}, mean={pnod_diff_DF.mean():.6f}")
    # 20秒以降の差分
    if min_pnod_DF > 200:
        print(f"  p_nod diff (D vs F) after 20s: max={pnod_diff_DF[200:].max():.6f}, mean={pnod_diff_DF[200:].mean():.6f}")

    # ===================================================================
    # Visualization
    # ===================================================================
    save_dir = os.path.join(_here, "visualizations", "compare_three_approaches")
    os.makedirs(save_dir, exist_ok=True)

    visualize_three_approaches(
        wav_user=wav_user,
        wav_erica=wav_erica,
        heads_A=heads_A,
        heads_B=heads_B,
        heads_C=heads_C,
        frame_hz=FRAME_RATE,
        sample_rate=SAMPLE_RATE,
        save_dir=save_dir,
        nod_threshold=0.5,
        heads_D=heads_D,
        heads_E=heads_E,
        heads_F=heads_F,
    )

    print("Done!")


if __name__ == "__main__":
    main()
