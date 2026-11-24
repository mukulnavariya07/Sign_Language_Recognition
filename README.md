# Sign Language Recognition — Static ASL Alphabet

CPU-friendly webcam demo classifying static American Sign Language alphabet hand shapes and typing stable letters into an on-screen bar. Educational prototype, not a certified accessibility or translation system.

## Problem
A camera-based hand-shape classifier can demonstrate one small component of assistive communication technology. This project recognizes static alphabet poses; it does not translate full sign-language conversations.

## Pipeline
```text
Webcam / labeled photos
  -> MediaPipe Tasks HandLandmarker (21 xyz landmarks)
  -> wrist-centering + wrist-to-MCP9 scale normalization
  -> 63 features -> labeled CSV
  -> train/test split (mirror training samples only)
  -> stratified CV: Random Forest / calibrated SVM / KNN / MLP
  -> refit best CV macro-F1 model -> held-out evaluation
  -> confidence filter -> vote smoothing -> hold-to-type UI
```

## Alphabet scope
A–Y **excluding J**. J and Z require motion and are not represented by a static-frame model. One hand is detected. No words, grammar, or continuous sign-language translation are inferred.

## Setup (Python 3.9+)
Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
MediaPipe wheel availability depends on Python/platform; use a supported Python version in a clean venv if installation fails. First detector run downloads `models/hand_landmarker.task` from the Google MediaPipe model URL in `src/config.py`. If download fails, save that model file at the same path manually.

## Run
1. Capture webcam data (press the target alphabet key to capture the default 100 valid frames; Q exits):
   `python -m src.collect_data --frames 100`
2. Or images in `image-root/A/*.jpg`, `image-root/B/*.jpg`, etc:
   `python -m src.extract_from_images image-root`
3. Train, compare, evaluate and save model/reports:
   `python -m src.train`
4. Live recognition:
   `python -m src.recognize`
   Hold a pose for ~0.5 seconds. SPACE deletes the last character, C clears, Q quits.
5. Tests: `pytest -q`

## Synthetic smoke test — FAKE data only
```bash
python -m src.synthetic
python -m src.train
pytest -q
```
Synthetic vectors are artificial, not hand landmarks. Never report their metrics as real recognition accuracy. Replace them with collected real data before evaluating.

## Results (fill with your own real data)
| Dataset / session | Best model | CV macro-F1 | Held-out accuracy | Held-out macro-F1 |
|---|---|---:|---:|---:|
| Real independent session | — | — | — | — |

## Limitations and responsible evaluation
- Static letters only (no J/Z motion signs), one hand, no language translation.
- Accuracy varies by signer, pose, camera, lighting, background, and collection quality.
- Adjacent frames are near-duplicates and can inflate random-split scores. Collect a separate session (preferably a different day/signer) and evaluate it independently. For stronger evaluation, split by session/signer before model selection.
- Mirroring negates x features to augment training, but does not guarantee all signs are mirror-equivalent.
- Inference speed depends on CPU and camera resolution.

## Extensions
Session-aware splits; temporal models for dynamic signs; consented diverse datasets; per-signer evaluation; text-to-speech; phrase editor; model profiling and optimized inference.

MIT License.
