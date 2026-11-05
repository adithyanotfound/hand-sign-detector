# Real-time sign recognition

Recognises 5 hand signs from a webcam, entirely on this Mac with no API calls. It draws the
hand skeleton, the sign's name and its confidence live, and builds a line of text from the
signs you hold.

## Run it

```sh
.venv/bin/python src/realtime.py
```

Press `q` or `Esc` to quit and `c` to clear the text. If the wrong camera opens (an iPhone, for
example), add `--camera 1`.

The first run needs camera permission. Allow it when macOS asks, then run the command again. If
it doesn't ask, go to System Settings > Privacy & Security > Camera, allow the app you're running
this from, and restart that app.

## How to make each sign

The model learned these specific poses from photos, so match them closely:

| Sign | Pose |
|---|---|
| Hello | Open palm facing the camera, fingers spread |
| I Love You | Thumb, index and pinky up; middle and ring fingers down; palm facing the camera |
| Peace | Index and middle up in a V; palm facing the camera |
| Thank You | Flat hand with fingers together, tilted, with the back of the hand to the camera (the data shows it held over the mouth) |
| Yes | **Horizontal** fist with the knuckles facing the camera. An upright fist won't trigger it |

Either hand works. Keep your whole hand in frame.

## How it works

```
webcam -> MediaPipe hand landmarks (21 points) -> normalise -> small neural net -> smoothing -> display
```

- **Landmarks:** Google's MediaPipe hand model (7.5 MB) finds the hand. It is used as-is.
- **Classifier, trained here:** a NumPy neural net with layers 42 -> 64 -> 32 -> 6, 23 KB in size.
  It sees only the hand's x/y shape, relative to the wrist and scaled to the hand's size, so
  background, skin tone and distance from the camera don't feed into it directly.
- **`(no sign)`:** a sixth output, trained on other hand shapes, so that not every hand gets named
  as a sign. A sign is shown only when its smoothed confidence reaches 60%.

## Performance

Measured on an M2 MacBook Air (8 GB RAM), playing a 30 FPS test video:

| | |
|---|---|
| Time per frame (hand tracking + classifier) | about 14 ms median, 26 ms at the 95th percentile |
| CPU | about 50% of one core |
| Memory | about 220 MB |
| Classifier alone | about 11 microseconds per prediction |
| Landmark extraction / full evaluation and training | 70 s / 30 s |

## Accuracy

Scores come from testing on a person left out of training. A random split would be misleading
because the dataset is burst photos of about 3 people, and near-identical frames would land on
both sides, giving a score near 100%. People were identified approximately, from outfit colours.

| Held-out result | |
|---|---|
| Signs named correctly | 92.6% (±1.0 over 5 training seeds) |
| Hello / Thank You / Yes | 100% (one seed) |
| I Love You | 98.8% (one seed) |
| Peace | 62.4% (one seed); see below |
| Never-seen hand shapes wrongly named as a sign | 13.3% (±1.2) |

Outside check, from different people and setting: the alphabet letter V, the same shape as Peace
and never trained on, is called Peace in 80% of 40 images.

### Limitations

- **This recognises static hand poses; it is not sign language translation.** Real ASL Hello,
  Thank You and Yes involve movement, and signing also uses the face, body and grammar. The model
  sees single poses and has no knowledge of the face, so Thank You doesn't need to be near the
  mouth.
- **The test set is small and uneven.** It covers about 3 people, and they aren't evenly spread
  across signs: Person A has no I Love You or Peace images, so those two are tested on persons B
  and C. The 100% scores cover few people, so expect lower results on new hands, lighting and
  angles.
- **Peace's score is partly pessimistic.** When Person B is held out, only 33 Peace images remain
  for training. The shipped model trains on all 250, which is more in line with the 80% outside
  check, but that check is also only 40 images.
- **Unfamiliar shapes can still be named as signs.** In testing, unseen W was called Peace and
  unseen F was called Hello. The shipped model has trained on both as "no sign", but other
  unfamiliar shapes can make the same mistake.
- **Only one hand at a time.** Tracking a second hand would make the palm detector run on every
  frame, which costs more CPU.
- **Not yet tested on a live camera.** The app was tested end to end on video files in a real
  window, but camera permission wasn't available in the build environment.

The dataset's `ok` folder is not used as a sign, because it shows two raised fingers rather than
the usual OK gesture. Those images train the `(no sign)` output instead.

## Rebuild from scratch

```sh
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -I src/download_data.py        # about 530 MB, checksums verified, no login
.venv/bin/python -I src/extract_landmarks.py    # about 70 s
.venv/bin/python -I src/train.py                # about 30 s; prints the evaluation
```

Results are saved in `models/report.json`. Retraining reproduces the same model file. Once
`data/landmarks.npz` exists, `data/raw/` (781 MB) is only needed to re-run extraction.

## Files

| File | Purpose |
|---|---|
| `src/realtime.py` | The webcam app |
| `src/train.py` | Evaluation and training |
| `src/extract_landmarks.py` | Turns images into cached landmarks |
| `src/download_data.py` | Downloads the data and the hand model |
| `src/features.py`, `src/model.py` | Feature code and classifier shared by training and the app |
| `models/sign_classifier.npz` | The trained classifier |

## Credits

Both datasets are CC BY 4.0, from Mendeley Data:

- Sign_Language_Dataset: <https://doi.org/10.17632/pwyrdzwzwf.1>. Signs and `(no sign)` examples.
- SignAlphaSet: <https://doi.org/10.17632/8fmvr9m98w.3>. A sample of its images, used for
  `(no sign)` examples and the outside check.

Hand landmarks come from [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
