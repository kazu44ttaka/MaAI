<h1>
<p align="center">
相槌予測モデル（タイミング）
</p>
</h1>
<p align="center">
README: <a href="vap_bc.md">English </a> | <a href="vap_bc_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータは `bc` に設定してください。

入力は2チャンネルの16kHz音声データが必要です（ch1がユーザ音声、ch2がシステム音声を想定）。
システムの相槌タイミングを予測します。

出力は相槌の事後確率 `p_bc` です。

</br>

## 対応言語

以下の言語に対応しています。
`Maai` クラスの `lang` パラメータで指定してください。

### 日本語（`lang=jp`）

本モデルは以下の日本語データセットで学習されています：
- [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191)

</br>

### 英語（`lang=en`）

本モデルは以下の英語データセットで学習されています：
- オンライン会話データセット

</br>

### 中国語（`lang=ch`）

本モデルは以下の中国語データセットで学習されています：
- オンライン会話データセット

</br>

### 3言語対応（日本語＋英語＋中国語）（`lang=tri`）

本モデルは以下の3言語データセットで学習されています：
- オンライン会話データセット（日本語、英語、中国語）

</br>

## 実装例

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="bc", lang="jp", frame_rate=10, context_len_sec=5, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai.start()

while True:
    result = maai.get_result()

    print(result['p_bc'])     # 相槌の確率
```

</br>

## パラメータ

利用可能なパラメータを以下にまとめます。
`frame_rate` はVAPモデルが1秒あたりに処理するサンプル数を指定します。
ご利用の計算環境に合わせて、この値を調整してください。

| `lang` | `model_type` | `frame_rate` |
| --- | --- | --- |
| jp | normal | 5, 10, 20 |
| jp | normal-ver2 | 12.5 |
| en | normal | 5, 10, 20 |
| en | normal-ver2 | 12.5 |
| ch | normal | 5, 10, 20 |
| ch | normal-ver2 | 12.5 |
| tri | normal | 5, 10, 20 |
| tri | normal-ver2 | 12.5 |

</br>

## 📚 論文・参考文献

このモデルを利用した成果を発表する際は、以下の論文を引用してください。🙏

Koji Inoue, Divesh Lala, Gabriel Skantze, Tatsuya Kawaharaa<br>
__Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection__<br>
https://aclanthology.org/2025.naacl-long.367/<br>

```
@inproceedings{inoue2025vapbc,
    author = {Koji Inoue and Divesh Lala and Gabriel Skantze and Tatsuya Kawahara},
    title = {Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection},
    booktitle = {Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
    pages = {7171--7181},
    year = {2025},
    url = {https://aclanthology.org/2025.naacl-long.367/},
}
```
