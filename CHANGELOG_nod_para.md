# dev/nod-para ブランチ変更内容

`origin/main` からの差分（コミット `032dbf4`）。16 ファイル変更、+3962 / -130 行。

## 概要

**nod_para モード**（うなずきパラメータ予測）の推論パイプラインを MaAI に追加。
訓練コード（`VAP_Nodding_para`）の `VAPModel` を、リアルタイム推論に適した形で MaAI フレームワークに移植した。

---

## 新規ファイル

### `src/maai/encoder_maai.py`（+941 行）

Distilled WavLM Student Model の推論用実装。`VAP_Nodding_para/custom_maai_enc.py` からの移植。

- `CNNFrontEnd`: WavLM 互換 7 層 CNN フロントエンド（50Hz 出力、causal zero-padding）
- `CausalSelfAttentionALiBi`: ALiBi 位置エンコーディング付き causal self-attention。KV キャッシュ対応
- `TransformerBlock`: Pre-LayerNorm + ALiBi attention + MLP
- `StudentModel`: CNN → Transformer (`blocks_50` + `blocks_low`) パイプライン。`forward_cnn` / `forward_transformer` で CNN と Transformer を分離実行可能
- `EncoderMaai`: StudentModel のラッパー。ダウンサンプリング（50Hz → 10Hz）対応
  - `from_state_dict()`: チェックポイントの state_dict からアーキテクチャ（`d_model`, `n_heads`, 層数, `mlp_ratio` 等）を自動推論して構築
  - `forward_cnn()`: CNN のみ実行（チャンク差分計算用）
  - `forward_transformer()`: Transformer + ダウンサンプルのみ実行

### `src/maai/models/vap_nod_para.py`（+325 行）

うなずきパラメータ予測モデル。`VAP_Nodding_para/model_nod_para.py` からの移植。

- GPT（`ar_channel` + `ar` の 2 段）、KV キャッシュ対応
- 出力ヘッド: `nod_count`（回数 3 クラス）, `nod_range`（振幅）, `nod_speed`（速度）, `nod_swing_up_binary`/`value`/`continuous`（振り上げ）
- z-score 逆正規化（`nod_param_stats` による denormalize）
- `encode_audio_cnn()` / `encode_audio_transformer()`: CNN と Transformer の分離呼び出し
- `load_encoder_from_state_dict()`: state_dict から EncoderMaai を構築し、`decrease_dimension` 層も自動設定
- 共有エンコーダー（`nod_shared_encoder`）、MLP ヘッド（`nod_head_mlp_*`）等のオプション対応

### `src/maai/models/config.py`（+100 行）

`VapConfig` dataclass。nod_para 用の設定フィールドを追加：

- `nod_count_binary`: 2 値 vs 3 クラス選択
- `nod_head_mlp_*`: 各タスクの MLP ヘッド有効化設定
- `nod_shared_encoder` / `nod_shared_encoder_dim`: 共有特徴抽出層
- `gpt_output_dropout`: GPT 出力の Dropout

### `example/nod_para/nod_para_mic.py`（+83 行）

マイク入力によるリアルタイムうなずきパラメータ予測のサンプルスクリプト。

### `example/nod_para/compare_three_approaches.py`（+1333 行）

6 つの推論アプローチ (A〜F) を比較する検証スクリプト。訓練コードとの一致性を検証。

### `example/nod/compare_kv_cache.py`（+320 行）

nod モードの KV キャッシュ有無を比較する検証スクリプト。

### `docs/debug_streaming_vs_fullseq.md`（+262 行）

ストリーミング推論とフルシーケンス推論の一致性デバッグ記録。

---

## 変更ファイル

### `src/maai/model.py`（+539 / -130 行）

`Maai` クラスに nod_para モードの推論パイプラインを追加。主な変更点:

#### チェックポイント読み込み（`__init__`）

- **`vap` パッケージ互換スタブ** (`_PermissiveModule`, `_setup_vap_compat`, `_cleanup_vap_compat`): 訓練コードで保存されたチェックポイントを `torch.load` で読み込むため、`vap.*` モジュールのスタブを `sys.modules` に登録。`__main__.OptConfig` 等も動的に生成
- **`_build_nod_para_config()`**: チェックポイントの `hyper_parameters` から `VapConfig` を構築
- **`n_heads` 自動推論**: `encoder.model.proj.weight` の shape から `d_model` を取得し、`d_model // 64`（head_dim=64）で推定
- **`nod_param_stats` 抽出**: チェックポイントの `callbacks` から denormalize 用統計量を取得

#### パラメータ追加

- `use_kv_cache: bool = True`: GPT の KV キャッシュ有効化
- `use_anchor_frames: bool = True`: アンカーフレームの使用有無

#### nod_para 推論パイプライン（`process()`）

- **CNN 差分計算**: `encode_audio_cnn()` で新チャンクのみ CNN を実行し、50Hz 特徴量バッファに蓄積
- **アンカーフレーム**: 最初のチャンクの CNN 出力（causal ゼロパディングにより特徴的な値を持つ先頭 5 フレーム）を保存。バッファトリミング後に Transformer 入力の先頭に再付加し、ALiBi の位置基準として機能させる
- **Transformer 全ウィンドウ再計算**: `encode_audio_transformer()` で最大 20 秒分の 50Hz バッファ全体を毎フレーム（0.1 秒ごと）再計算
- **GPT KV キャッシュ**: nod_para モードでは Transformer 出力の最終フレームのみを GPT に入力し、KV キャッシュで逐次推論。`_vap_seq_len - 1` でキャッシュ長を制限

#### 初回チャンク処理の修正

- 初回チャンク（`_first_chunk`）では `frame_contxt_padding` を 0 にして、学習コードと同じゼロパディング動作を再現

#### 結果出力

- `nod_para` 用の出力辞書を追加: `p_bc`, `p_nod`, `nod_count`, `nod_range`, `nod_speed`, `nod_swing_up_binary`, `nod_swing_up_value`, `nod_swing_up_continuous`

### `src/maai/output.py`（+126 / -4 行）

#### ConsoleBar

- `nod_count`: argmax で最大クラスを表示（「1回」「2回」「3回+」）
- `nod_range`: 0〜0.15 の範囲でバー表示
- `nod_speed`: 0〜0.25 の範囲でバー表示
- `p_bins` の 2 話者ネスト構造表示に対応
- `value` が数値でない場合の例外ハンドリング追加

#### GuiPlot

- `nod_count`: 3 クラスの確率を時系列ラインプロットで表示
- `nod_range`, `nod_speed`, `nod_swing_up_value`, `nod_swing_up_continuous`: 回帰値の時系列プロット（y 軸自動スケール）
- `nod_swing_up_binary`: 0〜1 の fill_between プロット

#### TcpReceiver / TcpTransmitter

- `nod_para` モードのシリアライズ/デシリアライズに対応

### `src/maai/util.py`（+88 行）

- `conv_vapresult_2_bytearray_nod_para()` / `conv_bytearray_2_vapresult_nod_para()`: nod_para 結果の TCP バイト列変換

### `src/maai/encoder_components.py`（+5 / -3 行）

- `load_CPC()`: `checkpoint_cpc` が空パスの場合に `makedirs` を呼ばないよう修正（`FileNotFoundError` 防止）
- `get_cnn_layer()`: `mode` パラメータ追加。`"cconv"` で `CConv1d`、デフォルト `"conv"` で `nn.Conv1d` を使用

### `src/maai/input.py`（+8 行）

- `Wav`: ステレオ WAV 入力時に 2ch 目を使用するよう修正。`audio_gain` パラメータ対応

### `src/maai/models/vap_bc.py`, `vap_bc_2type.py`, `vap_nod.py`, `vap_prompt.py`（各 +4 行）

- `GPTStereo` の `forward()` 呼び出しに `past_kv1_c`, `past_kv2_c`（クロスアテンション用 KV キャッシュ）を追加。キャッシュ辞書に `cross1_c`, `cross2_c` を保存

---

## アーキテクチャ

```
マイク入力 (16kHz)
    │
    ▼ 0.1秒チャンク
┌──────────────────────────┐
│  CNN フロントエンド        │  ← 差分計算（新チャンクのみ）
│  (WavLM アーキテクチャ)    │
└──────────┬───────────────┘
           │ 50Hz 特徴量
           ▼
┌──────────────────────────┐
│  50Hz バッファ (最大20秒)  │  ← アンカーフレーム付加
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  ALiBi Transformer       │  ← 毎回全ウィンドウ再計算
│  (8〜10層, causal)       │
└──────────┬───────────────┘
           │ 10Hz ダウンサンプル (stride-5 Conv1d)
           ▼
┌──────────────────────────┐
│  GPT (channel + cross)   │  ← KVキャッシュ可能
└──────────┬───────────────┘
           │
           ▼
  nod_count, nod_range, nod_speed, swing_up
```

---

## 検証結果

`compare_three_approaches.py` による 6 方式の比較:

| 比較 | 結果 |
|---|---|
| A (訓練コード) vs B (MaAI 一括) | 一致（浮動小数点精度内） |
| D (ストリーミング, KVキャッシュなし) vs F (訓練コード sliding window) | 一致 |
| E (ストリーミング, KVキャッシュあり) vs D | 微小な差（cos similarity > 0.999） |

### Encoder KV キャッシュの検討

StreamingLLM 方式のアンカー付き Encoder KV キャッシュを試験的に実装・検証した結果、10 層の Transformer では誤差蓄積が致命的（cos similarity ≈ 0.5）であり、速度向上もないため不採用とした。詳細は `docs/debug_streaming_vs_fullseq.md` を参照。
