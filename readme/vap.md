<h1>
<p align="center">
Turn-Taking (VAP) Model
</p>
</h1>
<p align="center">
README: <a href="vap.md">English </a> | <a href="vap_JP.md">Japanese (日本語) </a>
</p>

Set the `mode` parameter of the `Maai` class to `vap`.

The input should be two-channel 16kHz audio.

The outputs are `p_now`, which represents the probability of voice activity between two speakers in the next 0–600 milliseconds, and `p_future`, which corresponds to 600–2000 milliseconds in the future.
For typical turn-taking implementations, it is recommended to use `p_now`.
Both outputs are returned as dictionaries.

</br>

## Supported Languages

The following languages are supported.
Specify the language using the `lang` parameter of the `Maai` class.

### Japanese (`lang=jp`)

This model is trained on the following Japanese datasets:
- [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/)
- [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/)
- [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191)

</br>

### Japanese (MIT License) (`lang=jp_kyoto`)

This model is trained on the following Japanese dataset:
- [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191)

This model is also released under the MIT license.

</br>

### English (`lang=en`)

This model is trained on the following English dataset:
- [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)
- Online Conversation Dataset

</br>

### English (MIT License) (`lang=en_kyoto`)

This model is trained on the following English dataset:
- Online Conversation Dataset

This model is also released under the MIT license.

</br>

### Chinese (`lang=ch`)

This model is trained on the following Chinese dataset:
- [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)
- Online Conversation Dataset

</br>

### Chinese (MIT License) (`lang=ch_kyoto`)

This model is trained on the following Chinese dataset:
- Online Conversation Dataset

This model is also released under the MIT license.

</br>

### Tri-lingual (JPN + ENG + CHN) (`lang=tri`)

This model is trained on the following three-language datasets:
- [Switchboard corpus](https://catalog.ldc.upenn.edu/LDC97S62)
- [HKUST Mandarin Telephone Speech](https://catalog.ldc.upenn.edu/LDC2005S15)
- [Travel Agency Task Dialogue](https://aclanthology.org/2022.lrec-1.619/)
- [Human-Robot Dialogue](https://aclanthology.org/2025.naacl-long.367/)
- [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191)

</br>

### Tri-lingual (JPN + ENG + CHN, MIT License) (`lang=tri_kyoto`)

This model is trained on the following three-language dataset:
- [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191)

This model is also released under the MIT license.

</br>

## Example

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

## Parameters

The available parameters are summarized below.
`model_type` selects the model variant.

- `"normal"`: the existing model variant used in previous releases
- `"normal-ver2"`: a new model variant that uses Mimi as the encoder

`frame_rate` specifies the number of samples processed per second by the VAP model.
Please adjust this value according to your computing environment.

For `model_type="normal"`, the available `frame_rate` values are shown in the table below.
For `model_type="normal-ver2"`, only `frame_rate=12.5` is supported.

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

</br>

## 📚 Publication

Please cite the following paper, if you made any publications made with this model. 🙏

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

If you use the multi-lingual VAP model, please also cite the following paper.

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
