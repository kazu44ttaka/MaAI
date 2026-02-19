# compare_three_approaches.py — nod_para モード 6 方式検証

## 概要

`nod_para` モード（うなずきパラメータ予測）の推論パイプラインを 6 つの異なるアプローチで実行し、出力の一致性と処理速度を検証するスクリプト。訓練コード（`VAP_Nodding_para`）と MaAI 推論コードの間に差異がないことを確認する目的で作成した。

## 実行方法

```bash
cd MaAI
uv run python example/nod_para/compare_three_approaches.py
```

## 比較する 6 つのアプローチ

| アプローチ | パイプライン | エンコーダー | GPT | KV キャッシュ |
|---|---|---|---|---|
| **A** | 訓練コード `VAPModel.forward()` | MaAI Encoder (フルシーケンス) | 訓練コードの GPT | なし |
| **B** | MaAI `encode_audio()` + `forward()` 一括 | MaAI Encoder (フルシーケンス) | MaAI GPT | なし |
| **C** | MaAI 疑似ストリーミング（チャンク蓄積） | MaAI Encoder (CPC KV キャッシュ) | MaAI GPT (蓄積して毎回全体) | なし |
| **D** | `Maai` クラス `process()` ループ | MaAI Encoder (CNN差分 + Transformer全再計算) | MaAI GPT | **なし** |
| **E** | `Maai` クラス `process()` ループ | MaAI Encoder (CNN差分 + Transformer全再計算) | MaAI GPT | **あり** |
| **F** | 訓練コード 20 秒スライディングウィンドウ | MaAI Encoder (フルシーケンス/毎回) | 訓練コードの GPT | なし |

### 目標

```
A ≈ B ≈ C ≈ D ≈ E ≈ F （浮動小数点精度の範囲内で一致）
```

### `RUN_ABC` フラグ

- `RUN_ABC = True`: A, B, C, D, E, F の全 6 方式を実行
- `RUN_ABC = False`（デフォルト）: D, E, F のみ実行（高速）

A, B, C は訓練コード（`VAP_Nodding_para/vap/`）の import が必要。

## 出力ヘッド

各アプローチで以下のヘッド値を収集し比較する：

| ヘッド | 説明 | 値の型 |
|---|---|---|
| `p_nod` | うなずきタイミング確率 | float (0〜1)、sigmoid |
| `nod_count` | うなずき回数 | argmax（0=1回, 1=2回, 2=3回+）|
| `nod_range` | うなずき振幅 | float（z-score 逆正規化後） |
| `nod_speed` | うなずき速度 | float（z-score 逆正規化後） |
| `nod_swing_up` | 振り上げ有無 | float (0〜1)、sigmoid |

## 各アプローチの実装詳細

### A: `run_training_forward()`

訓練コードの `VAPModel` をそのまま使用。WAV 全体を `(B=1, C=2, T)` のテンソルとして一括入力。全フレームの全ヘッド出力を取得。

**チャンネル規約**: ch0 = ERICA (speaker 0)、ch1 = User (speaker 1)

### B: `encode_like_maai_full()` + `run_gpt_heads()`

MaAI の `VapGPT_nod_para` を使用。WAV 全体を `encode_audio(cache=None)` で一括エンコードし、GPT + 全ヘッドを実行。A と B の差分は「訓練コードの VapGPT」と「MaAI の VapGPT_nod_para」の実装差のみ。

### C: `run_pseudo_streaming()`

MaAI の `VapGPT_nod_para` を疑似ストリーミングで使用。

- 1600 サンプル（0.1 秒）ずつチャンクを投入
- 初回チャンク: `encoder_cache=None`、`n_skip=0`
- 以降: 640 サンプルオーバーラップ、`n_skip=2`、CPC エンコーダー KV キャッシュ使用
- GPT はフレーム蓄積＋毎回全体 forward（KV キャッシュなし）

### D: `run_maai_class(use_kv_cache=False)`

本番の `Maai` クラスを `mode='nod_para'` でインスタンス化し、`process()` にチャンクを直接投入。

- ダミー入力（`_DummyInput`）を使用して `Maai.__init__()` を通す
- 1600 サンプルずつ `process()` を呼び出し
- `result_dict_queue` から結果を回収
- **GPT KV キャッシュなし**: 毎フレーム Encoder 出力全体を GPT に入力

nod_para の Encoder パイプライン:
1. CNN フロントエンド: 新チャンクのみ差分計算
2. 50Hz 特徴量バッファに蓄積（最大 20 秒 = 1000 フレーム）
3. アンカーフレーム付加（`USE_ANCHOR_FRAMES=True` の場合）
4. Transformer 全ウィンドウ再計算（毎フレーム）
5. 10Hz ダウンサンプル

### E: `run_maai_class(use_kv_cache=True)`

D と同じだが **GPT KV キャッシュあり**。Encoder Transformer 出力の最終フレームのみを GPT に入力し、過去フレームは KV キャッシュで保持。

### F: `run_sliding_window_forward()`

訓練コードの `VAPModel` を 20 秒スライディングウィンドウで使用。

- 各フレームで `[max(0, t-20s), t]` の音声ウィンドウを切り出し
- `VAPModel(waveform=...)` でフルシーケンス推論
- 最終フレームの出力値を収集

D と同じ時間ウィンドウを訓練コードで処理することで、「D の Encoder + GPT」が「訓練コードの Encoder + GPT」と一致するかを検証。

## チェックポイント読み込み

### 訓練コード用（A / F）: `load_model_training()`

1. `VAP_Nodding_para/vap/train_nod_para.py` の `VAPModel` を import
2. `torch.load` で `__main__.OptConfig` 等が必要なため `__main__` にクラスを登録
3. `conf.maai_model_pt`（エンコーダーチェックポイント）のパスをローカルに差し替え（`MaAI/assets/` → `VAP_Nodding_para/` の順で検索）
4. `VAPModel.load_from_checkpoint()` で読み込み
5. `nod_param_stats`（denormalize 用統計量）を取得

### MaAI 用（B / C）: `load_model_maai()`

1. `model.py` の `_setup_vap_compat()` で vap スタブモジュールを登録
2. `torch.load` でチェックポイント読み込み
3. state_dict から `model.` prefix を除去
4. `_build_nod_para_config()` で `VapConfig` を構築
5. `n_heads` を state_dict から自動推論（`d_model // 64`）
6. `VapGPT_nod_para` を構築し `load_encoder_from_state_dict()` でエンコーダーを設定
7. `nod_param_stats` を `callbacks` から取得

## 可視化

`visualizations/compare_three_approaches/` に以下を出力：

### 全体比較 (`all_heads_comparison.png`)

6 段のサブプロット:
1. 波形（User + ERICA）
2. `p_nod`: 全アプローチ重畳 + 閾値ライン + 閾値超え区間のハイライト
3. `nod_count`: ステッププロット（1回 / 2回 / 3回+）
4. `nod_range`: 時系列プロット
5. `nod_speed`: 時系列プロット
6. `nod_swing_up`: 時系列プロット + 0.5 閾値ライン

タイトルに各アプローチ間の `p_nod` 最大差分を表示。

### 個別ヘッド (`{head_key}_timeseries.png`)

各ヘッドについて 2 段のサブプロット:
- 上段: 全アプローチ重畳
- 下段: 差分プロット（A vs C/D/E/F、または D vs E/F）

## コンソール出力

### 各アプローチの統計量

```
p_nod frames: 400
p_nod: min=0.0012, max=0.9876, mean=0.3456
Frames > 0.5: 120
Time: 12.345s, RTF: 0.3086
```

### RTF サマリ

各アプローチの処理時間と Real-Time Factor（RTF = 処理時間 / 音声長）を表示。RTF < 1.0 でリアルタイム処理可能。

### 比較サマリ

全アプローチペアの `p_nod` 最大差分・平均差分。

### B vs C エンコーダー段階別比較（`RUN_ABC=True` 時）

`RUN_ABC=True` の場合、B（フルシーケンス）と C（チャンク＋KV キャッシュ）のエンコーダー出力を以下の 3 段階で比較:
1. **50Hz CNN+proj**: CNN フロントエンド + 線形射影の出力
2. **50Hz norm**: Transformer + LayerNorm の出力
3. **10Hz downsample**: 最終ダウンサンプル後の出力

各段階でフレームごとの最大差分を表示し、誤差の発生箇所を特定可能。

## 設定パラメータ

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `RUN_ABC` | `False` | True: A/B/C も実行、False: D/E/F のみ |
| `CHECKPOINT` | `MaAI/assets/medium_epoch17-...ckpt` | チェックポイントファイルパス |
| `WAV_FILE` | `example/wav_sample/jpn_sumida_16k.wav` | 入力 WAV ファイル |
| `DEVICE` | `cuda` / `cpu` | 推論デバイス（自動検出） |
| `FRAME_RATE` | 10 | フレームレート (Hz) |
| `CROP_SEC` | 40.0 | クロップ長 (秒) |
| `CROP_OFFSET_SEC` | 7.5 | クロップ開始オフセット (秒) |
| `USE_ANCHOR_FRAMES` | `True` | アンカーフレームの使用有無 |

## 期待される結果

| 比較 | 期待される差分 | 説明 |
|---|---|---|
| A vs B | ≈ 0 | 訓練コードと MaAI の forward が同一であることを確認 |
| B vs C | ≈ 0 | CPC エンコーダー KV キャッシュが正確であることを確認 |
| D vs F | ≈ 0 | 本番 `process()` ループが学習コードと同一結果を出すことを確認 |
| D vs E | < 0.001 | GPT KV キャッシュによる微小な差分のみ |
| A vs D | 差分あり | A はフルシーケンス、D は 20 秒ウィンドウなので 20 秒以前は異なる |
