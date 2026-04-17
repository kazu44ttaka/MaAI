<h1>
<p align="center">
Nod Parameter Prediction Model
</p>
</h1>
<p align="center">
README: <a href="vap_nod_para.md">English </a> | <a href="vap_nod_para_JP.md">Japanese (日本語) </a>
</p>

Set the `mode` parameter of the `Maai` class to `nod_para`, and set `model_type` to `normal-ver2`.

This model takes 2-channel 16 kHz audio as input, assuming ch1 as user audio and ch2 as system audio.

It predicts nod timing and the following kinematic parameters.

<p align="center">

| `parameter` | `description` |
| --- | --- |
| range | Nod amplitude (rad) |
| speed | Nod angular speed (rad/s) |
| repetitions | Number of nods (1, 2, 3+) |
| swing-up | Presence of an upward swing (0, 1) |

</p>

</br>

## Motion synthesis

Rule-based motion generation from these parameters is implemented in `src/maai/util.py` (e.g. `generate_natural_nod`). You can use it to synthesize nodding motion as in `example/nod/nod_generate_motion.py`.

<img src="../img/nod_motion_sample.png" width="800">
<img src="../img/nod_motion_sample.gif" width="300">

</br>

## Supported Languages

Currently, only Japanese is supported.  
Specify it with the `lang` parameter of the `Maai` class.

### Japanese (`lang=jp`)

This model is trained on the following Japanese dataset:
- [Human-Robot Dialogue Corpus]()

</br>

## Example Implementation

```python
from maai import Maai, MaaiInput

mic = MaaiInput.Mic(mic_device_index=0)
zero = MaaiInput.Zero()

maai = Maai(mode="nod_para", lang="jp", frame_rate=12.5, audio_ch1=mic, audio_ch2=zero, device="cpu", model_type="normal-ver2")
maai.start()

while True:
    result = maai.get_result()

    print(result['p_nod'])
    print(result['nod_repetitions'])
    print(result['nod_repetitions_pred'])
    print(result['nod_range'])
    print(result['nod_speed'])
    print(result['nod_swing_up'])
    print(result['nod_swing_up_pred'])
```

`nod_repetitions_pred` and `nod_swing_up_pred` are discrete predictions using thresholds tuned on validation data.

</br>

## Parameters

The available settings are summarized below.  
`frame_rate` is the number of frames the VAP model processes per second; adjust it for your hardware.

| `lang` | `frame_rate` |
| --- | --- |
| jp | 12.5 |
