# MaAI ストリーミング推論とフルシーケンス推論の一致性デバッグ

## 概要

MaAI の `EncoderMaai` を用いたストリーミング推論（チャンク逐次処理）が、
フルシーケンス推論（20秒分を一括処理）および訓練コードの推論結果と大きく乖離していた問題を調査・修正した。

### 検証フレームワーク

5つのアプローチを比較する検証スクリプト `compare_three_approaches.py` を作成：

| アプローチ | 説明 |
|---|---|
| **A** | 訓練コード (`VAP_Nodding_para/vap/train_nod_para.py`) の `VAPModel` で推論 |
| **B** | MaAI の `VapGPT_nod_para` で 20秒分を一括処理（`cache=None`） |
| **C** | MaAI の `VapGPT_nod_para` で疑似ストリーミング（0.1秒チャンク＋エンコーダKVキャッシュ） |
| **D** | `Maai` クラス経由（本番 `process()` ループ、GPT KVキャッシュなし） |
| **E** | `Maai` クラス経由（本番 `process()` ループ、GPT KVキャッシュあり） |

目標: A ≈ B ≈ C ≈ D ≈ E（浮動小数点精度の範囲内で一致）

---

## 発見された問題と修正

### 問題 1: A vs B — `n_heads` 推定の不一致

**症状**: `p_nod diff (A vs B): max=0.29`

**原因**: `EncoderMaai.from_state_dict()` が Transformer の `n_heads` を
`d_model // 32 = 512 // 32 = 16` とヒューリスティックに推定していたが、
訓練時のチェックポイントでは `n_heads=8` で保存されていた。
ヘッド数が異なると、同じ重み行列でも Attention の分割が変わるため出力が大きく異なる。

**修正** (`encoder_maai.py`):
```python
# from_state_dict() に明示的な n_heads パラメータを追加
def from_state_dict(cls, state_dict, frame_hz=50, lim_context_sec=-1, n_heads=None):
    if n_heads is None:
        n_heads = max(1, d_model // 32)  # フォールバック
```

### 問題 2: B vs C — CConv1d ダウンサンプルのゼロパディング（主因）

**症状**: `p_nod diff (B vs C): max=0.52`, `Downsample diff: max=3.56`

**原因**: 50Hz → 10Hz のダウンサンプルに使われる `CConv1d`（kernel=5, stride=5）は、
入力の左側に `kernel_size - 1 = 4` 個のゼロをパディングする。
フルシーケンスでは先頭フレームのみゼロパディングの影響を受けるが、
ストリーミングでは **毎チャンク** ゼロパディングされるため、
前チャンクのコンテキストが完全に失われていた。

```
フルシーケンス: frame k = conv([x_{5k-4}, x_{5k-3}, x_{5k-2}, x_{5k-1}, x_{5k}])
ストリーミング: frame k = conv([0, 0, 0, 0, x_{5k}])  ← 前の4フレームが欠落
```

**修正** (`encoder_maai.py` の `EncoderMaai.forward()`):
- キャッシュに `ds_buf` を追加し、各チャンクの最後の 4 フレーム（50Hz）を保存
- 次チャンクで `ds_buf` を前置して `F.conv1d` を直接呼び出し（`CConv1d` のゼロパディングをバイパス）

```python
if ds_buf is not None:
    z_with_ctx = torch.cat([ds_buf, z], dim=1)  # (B, 4+5, D)
    x = z_with_ctx.transpose(1, 2)
    # CConv1d のゼロパディングを使わず、実コンテキストで Conv1d
    x = F.conv1d(x, cconv.weight, cconv.bias, stride=cconv.stride, dilation=cconv.dilation)
    ...
else:
    # 最初のチャンク: ゼロパディングが正しい（フルシーケンスと同じ）
    z = self.downsample(z)
```

### 問題 3: B vs C — CNN フロントエンドの受容野カバー不足（副因）

**症状**: `CNN diff: max=0.01`（小さいが Transformer で増幅される）

**原因**: CNN フロントエンド（7層 Conv1d, WavLM 由来）の受容野は **400 サンプル**、
ストライドは **320 サンプル**（= 1 フレーム @ 50Hz）。
`frame_contxt_padding=320` では 1 フレーム分のオーバーラップしかなく、
受容野に対して **80 サンプル不足** していた。

```
受容野 = 400 samples
オーバーラップ = 320 samples (1 frame)
不足 = 80 samples → 先頭フレームの出力が不正確
```

**修正** (`model.py`):
```python
# nod_para モード: CNN 受容野をカバーする十分なオーバーラップ
self.frame_contxt_padding = 640  # 2 × 320 = 2 CNN frames
self.encoder_n_skip = 2          # 2フレーム分をスキップ
```

CPC エンコーダを使うモード（`vap`, `bc`, `nod` 等）は元の値を維持：
```python
# CPC モード: 元のオーバーラップ
self.frame_contxt_padding = 320
self.encoder_n_skip = 1
```

### 問題 4: D vs E — クロス注意 KV キャッシュの欠落

**症状**: `p_nod diff (D vs E): max=0.151`（KV キャッシュあり/なしで大きく乖離）

**原因**: `VapGPT_nod_para.forward()` が `GPTStereo` を呼ぶ際、
自己注意の KV キャッシュ (`past_kv1`, `past_kv2`) は保存・復元していたが、
**クロス注意**の KV キャッシュ (`past_kv1_c`, `past_kv2_c`) は保存していなかった。

そのため、KV キャッシュモードでは各フレームのクロス注意が
**現在のフレームの相手話者データのみ**を参照し、過去の相手話者フレームが見えていなかった。

```python
# 修正前: クロス注意キャッシュが欠落
out = self.ar(o1["x"], o2["x"],
    past_kv1=cache.get("cross1"),
    past_kv2=cache.get("cross2"),
    # past_kv1_c, past_kv2_c が渡されていない → 常に None
)

# 修正後: クロス注意キャッシュも保存・復元
out = self.ar(o1["x"], o2["x"],
    past_kv1=cache.get("cross1"),
    past_kv2=cache.get("cross2"),
    past_kv1_c=cache.get("cross1_c"),
    past_kv2_c=cache.get("cross2_c"),
)
new_cache = {
    ...
    "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
    "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
}
```

---

## 修正後の結果

| 指標 | 修正前 | 修正後 |
|---|---|---|
| p_nod diff (A vs B) | max=0.290 | **max=0.000** |
| p_nod diff (B vs C) | max=0.520 | **max=0.000152** |
| p_nod diff (A vs C) | max=0.520 | **max=0.000152** |
| p_nod diff (A vs D) | — | **max=0.000152** |
| p_nod diff (C vs D) | — | **max=0.000** |
| p_nod diff (A vs E) | max=0.151 | **max=0.000116** |
| p_nod diff (D vs E) | max=0.151 | **max=0.000062** |
| Downsample diff (B vs C) | max=3.56 | **max=0.0025** |
| CNN diff (B vs C) | 境界で大 | **max=0.00057** |

残差（~1e-4）は浮動小数点演算の精度限界によるもの。

### 問題 5: D/E — エンコーダ KV キャッシュなしモードで 20 秒以降に予測が崩壊

**症状**: `use_encoder_kv_cache=False` 設定の D/E で、20 秒（コンテキスト窓の上限）
を超えた時点から予測値が F（訓練コードによるスライディングウィンドウ推論）と大きく乖離し、
Transformer の 50Hz 出力（z50）が全ポジションで同一の定数値に退化する。

**原因**: CNN の因果パディング（causal zero-padding）が生成する先頭フレーム（L2≈10.17）は、
Transformer の ALiBi attention にとって位置参照の**アンカー**として機能していた。

- **F**（訓練コード）: 毎フレーム、スライディングウィンドウの音声を CNN に通すため、
  常にゼロパディング由来のアンカーフレームが先頭に存在する。
- **D/E**: CNN 特徴量をバッファに蓄積し、Transformer のみ再実行する方式。
  20 秒を超えるとバッファがスライドし、初回チャンクで生成されたアンカーフレームが
  追い出される。

アンカー消失後、全 CNN 入力が均一（L2≈4.5–4.6）になり、ALiBi attention が
位置を区別できなくなって全出力が定数に崩壊する。

```
正常時（20秒以前）:
  cnn_buf = [anchor(L2≈10.17), frame1, frame2, ..., frame999]
  → z50: 正常な位置依存の出力

崩壊時（20秒以降、アンカー消失）:
  cnn_buf = [frame501, frame502, ..., frame1500]  ← 全て L2≈4.5–4.6
  → z50: 全ポジションで [1.53, 0.79, 0.24, 0.36] に退化
```

**修正** (`model.py` の `Maai.__init__()` および `process()`):

1. **アンカーフレーム保持**: 初回チャンクの CNN 出力から先頭 5 フレーム
   （= 1 つの 10Hz ダウンサンプルブロック分）を `_cnn_anchor_ch1/ch2` に保存。

2. **条件付きプリペンド**: バッファが初めてトリムされた時点
   （`_cnn_buf_trimmed = True`）以降、Transformer 実行前にアンカーフレームを
   バッファの先頭に追加。トリム前はバッファ自体にアンカーが含まれているため不要。

3. **シーケンス長の一致**: バッファ最大長を `1000 - 5 = 995` に設定し、
   anchor(5) + buffer(995) = 1000 = F と同一のシーケンス長にする。
   これにより ALiBi の位置バイアスパターンが F と完全に一致。

4. **出力トリミング**: Transformer + downsample 後、出力が `_vap_seq_len`（200）
   を超える場合は末尾 200 フレームのみ採用。

```python
# __init__
self._cnn_anchor_frames = 5  # 保存するアンカーフレーム数
self._max_cnn_feat_frames = int(context_len_sec * 50) - self._cnn_anchor_frames  # 995

# process (初回チャンク)
if _first_chunk:
    self._cnn_anchor_ch1 = cnn1_new[:, :5, :].detach().clone()

# process (バッファトリム後)
if self._cnn_buf_trimmed and self._cnn_anchor_ch1 is not None:
    cnn1_for_tf = torch.cat([self._cnn_anchor_ch1, cnn1_cat], dim=1)  # [1, 1000, 512]

# process (出力トリム)
if e1.shape[1] > self._vap_seq_len:
    e1 = e1[:, -self._vap_seq_len:, :]  # 末尾200フレームのみ
```

**残存する差異**:
D/E のアンカーフレームは初回チャンクの固定値だが、F は毎回フレッシュに再計算する。
ただし、アンカーフレームの値は音声内容にほぼ依存しない定数
（L2 norm: 全ウィンドウで 10.166–10.167）であるため、この差異は極めて小さく、
ALiBi の距離減衰により予測値への影響は無視できる。
これは CNN 特徴量バッファリング方式の本質的な限界であり、
完全一致には毎フレーム 20 秒の生音声から CNN を再実行する必要がある（VRAM・RTF の制約上非現実的）。

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `MaAI/src/maai/encoder_maai.py` | `from_state_dict()` に `n_heads` パラメータ追加、`forward()` にダウンサンプルキャッシュ + `n_skip_frames` パラメータ追加 |
| `MaAI/src/maai/models/vap_nod_para.py` | `encode_audio()` で `n_skip_frames` をパススルー、`load_encoder_from_state_dict()` で `n_heads` をパススルー、`forward()` でクロス注意 KV キャッシュを保存・復元 |
| `MaAI/src/maai/model.py` | `frame_contxt_padding` を nod_para=640 / CPC=320 に分岐、`encoder_n_skip` 追加、`process()` で `n_skip_frames` を渡す、エンコーダロード時に `n_heads` を明示的に渡す、エンコーダ KV キャッシュなしモードでのアンカーフレーム保持・プリペンド・出力トリミング |

---

## 教訓

1. **ヒューリスティックな推定は危険**: `n_heads = d_model // 32` のような推定は、
   実際のチェックポイント値と一致しない場合がある。可能な限り明示的に渡す。
   注: 訓練コードの `StudentModel` のデフォルトは `n_heads=8` だが、
   `VapConfig` に含まれず `hyper_parameters` にも保存されないため、
   本番ロードパス (`model.py`) でもデフォルト 8 を明示的に渡す必要がある。

2. **Causal Conv1d のストリーミング対応には専用キャッシュが必要**:
   `CConv1d` は因果的パディング（左側にゼロ）を行うが、
   ストリーミングでは前チャンクの出力をキャッシュして置き換える必要がある。

3. **CNN 受容野とオーバーラップの整合性**:
   オーバーラップ量は CNN の受容野以上でなければ、境界フレームの出力が不正確になる。
   受容野の計算: `RF_l = RF_{l-1} + (k_l - 1) × stride_product_{l-1}`

4. **KV キャッシュは全注意層のキャッシュを保存する必要がある**:
   `GPTStereo` のように自己注意とクロス注意の両方を持つモデルでは、
   クロス注意の KV キャッシュも保存・復元しないと、
   相手話者の過去フレーム情報が失われる。

5. **CNN の因果パディングは暗黙の位置情報を生む**:
   Causal CNN のゼロパディングにより先頭フレームは特異な値（L2≈10.17）を持ち、
   ALiBi Transformer はこれを位置参照のアンカーとして利用している。
   特徴量バッファリング方式でバッファがスライドすると、このアンカーが失われ
   Transformer の出力が全ポジションで定数に退化する。
   対策: アンカーフレームを保存し、バッファのスライド後も常にプリペンドする。
