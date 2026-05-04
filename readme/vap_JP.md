<h1>
<p align="center">
ターンテイキング (VAP) モデル
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

`Maai` クラスの `mode` パラメータは `vap` に設定してください。

入力は2チャンネルの16kHz音声データが必要です。
出力は2つあり、`p_now` は2話者間の音声活動が次の0～600ミリ秒で発生する確率、`p_future` は600～2000ミリ秒先の確率を表します。
一般的なターンテイキング用途では `p_now` の利用を推奨します。
どちらの出力も辞書型で返されます。

</br>

## 対応言語

以下の言語に対応しています。
`Maai` クラスの `lang` パラメータで指定してください。

### 日本語（`lang=jp`）

本モデルは以下の日本語データセットで学習されています：
- [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)
- [ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)
- [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191)

</br>

### 日本語MITライセンス（`lang=jp_kyoto`）

本モデルは以下の日本語データセットで学習されています：
- [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191)

また、このモデルはMITライセンスで公開されています。

</br>

### 英語（`lang=en`）

本モデルは以下の英語データセットで学習されています：
- [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)
- オンライン会話データセット

</br>

### 英語MITライセンス（`lang=en_kyoto`）

本モデルは以下の英語データセットで学習されています：
- オンライン会話データセット

また、このモデルはMITライセンスで公開されています。

</br>

### 中国語（`lang=ch`）

本モデルは以下の中国語データセットで学習されています：
- [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)
- オンライン会話データセット

</br>

### 中国語MITライセンス（`lang=ch_kyoto`）

本モデルは以下の中国語データセットで学習されています：
- オンライン会話データセット

また、このモデルはMITライセンスで公開されています。

</br>

### 3言語対応（日本語＋英語＋中国語）（`lang=tri`）

本モデルは以下の3言語データセットで学習されています：
- [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)
- [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)
- [旅行代理店タスク対話コーパス](https://aclanthology.org/2022.lrec-1.619/)
- [ヒューマンロボット対話コーパス](https://aclanthology.org/2025.naacl-long.367/)
- [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191)

</br>

### 3言語対応MITライセンス（日本語＋英語＋中国語）（`lang=tri_kyoto`）

本モデルは以下の3言語データセットで学習されています：
- [オンライン会話データセット](https://www.arxiv.org/abs/2506.21191)

また、このモデルはMITライセンスで公開されています。

</br>

## 実装例

```python
from maai import Maai, MaaiInput

wav1 = MaaiInput.Wav(wav_file_path="path_to_your_user_wav_file")
wav2 = MaaiInput.Wav(wav_file_path="path_to_your_system_wav_file")

maai = Maai(mode="vap", lang="jp", frame_rate=10, audio_ch1=wav1, audio_ch2=wav2, device="cpu")

maai.start()

while True:
    result = maai.get_result()

    print(result['p_now'])
    print(result['p_future'])
```

</br>

## パラメータ

利用可能なパラメータを以下にまとめます。
`model_type` はモデル種別を指定します。

- `"normal"`: これまでのリリースで使っていた既存モデル
- `"normal-ver2"`: Mimiをエンコーダとして使用する新しいモデル

`frame_rate` はVAPモデルが1秒あたりに処理するサンプル数を指定します。
ご利用の計算環境に合わせて、この値を調整してください。

`model_type="normal"` の場合に利用可能な `frame_rate` は下表のとおりです。
`model_type="normal-ver2"` の場合は `frame_rate=12.5` のみ対応しています。

| `lang` | `model_type` | `frame_rate` |
| --- | --- | --- |
| jp | normal | 5, 10, 20 |
| jp | normal-ver2 | 12.5 |
| jp_kyoto | normal | 5, 10, 20 |
| jp_kyoto | normal-ver2 | 12.5 |
| en | normal | 5, 10, 20 |
| en | normal-ver2 | 12.5 |
| en_kyoto | normal | 5, 10, 20 |
| en_kyoto | normal-ver2 | 12.5 |
| ch | normal | 5, 10, 20 |
| ch | normal-ver2 | 12.5 |
| ch_kyoto | normal | 5, 10, 20 |
| ch_kyoto | normal-ver2 | 12.5 |
| tri | normal | 5, 10, 20 |
| tri | normal-ver2 | 12.5 |
| tri_kyoto | normal | 5, 10, 20 |
| tri_kyoto | normal-ver2 | 12.5 |

<br>

## 📚 論文・参考文献

このモデルを利用した成果を発表する際は、以下の論文を引用してください。🙏

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection__<br>
International Workshop on Spoken Dialogue Systems Technology (IWSDS), 2024<br>
https://arxiv.org/abs/2401.04868<br>

```
@inproceedings{inoue2024iwsds,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Real-time and Continuous Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {International Workshop on Spoken Dialogue Systems Technology (IWSDS)},
    year = {2024},
    url = {https://arxiv.org/abs/2401.04868},
}
```

トリリンガルVAPモデルを利用する場合は、以下も引用してください。

Koji Inoue, Bing'er Jiang, Erik Ekstedt, Tatsuya Kawahara, Gabriel Skantze<br>
__Multilingual Turn-taking Prediction Using Voice Activity Projection__<br>
Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pages 11873-11883, 2024<br>
https://aclanthology.org/2024.lrec-main.1036/<br>

```
@inproceedings{inoue2024lreccoling,
    author = {Koji Inoue and Bing'er Jiang and Erik Ekstedt and Tatsuya Kawahara and Gabriel Skantze},
    title = {Multilingual Turn-taking Prediction Using Voice Activity Projection},
    booktitle = {Proceedings of the Joint International Conference on Computational Linguistics and Language Resources and Evaluation (LREC-COLING)},
    pages = {11873--11883},
    year = {2024},
    url = {https://aclanthology.org/2024.lrec-main.1036/},
}
```
