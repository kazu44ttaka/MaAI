<h1>
<p align="center">
Backchannel Prediction Model (Timing)
</p>
</h1>
<p align="center">
README: <a href="vap_bc.md">English </a> | <a href="vap_bc_JP.md">Japanese (日本語) </a>
</p>

Set the `mode` parameter of the `Maai` class to `bc`.

This model takes 2-channel 16kHz audio data as input, assuming ch1 as user audio and ch2 as system audio.
It predicts system backchannel timings.

The output consists of posterior probability for backchannel `p_bc`.

</br>

## Supported Languages

Currently, only Japanese is supported.
Specify this with the `lang` parameter of the `Maai` class.

### Japanese (`lang=jp`)

This model is trained on the following Japanese dataset:
- [Online Conversation Dataset](https://www.arxiv.org/abs/2506.21191)

</br>

### English (`lang=en`)

This model is trained on the following English dataset:
- Online Conversation Dataset

</br>

### Chinese (`lang=ch`)

This model is trained on the following Chinese dataset:
- Online Conversation Dataset

</br>

### Multilingual (`lang=tri`)

This model is trained on the following multilingual dataset:
- Online Conversation Dataset (Japanese, English, Chinese)

</br>

## Implementation Example

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="bc", lang="jp", frame_rate=10, context_len_sec=5, audio_ch1=mic, audio_ch2=zero, device="cpu")
maai.start()

while True:
    result = maai.get_result()

    print(result['p_bc'])     # Probability of backchannel
```

</br>

## Parameters

The available parameters are summarized below.
`frame_rate` specifies the number of samples processed per second by the VAP model.
Please adjust this value according to your computing environment.

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

## 📚 Papers & References

When publishing results using this model, please cite the following paper. 🙏

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
