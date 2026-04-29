# ASL Finger Spelling Recogniser

Real-time American Sign Language (ASL) finger spelling recognition using a trained SVM classifier, MediaPipe hand landmarks, and ElevenLabs text-to-speech output.

## What it does

- Detects hand landmarks in real time via webcam using MediaPipe
- Classifies ASL letter signs using a Support Vector Machine (SVM) trained on custom-collected data
- Builds up a word letter-by-letter as you hold each sign
- Speaks the completed word aloud using the ElevenLabs TTS API
- Confidence-based hold bar UI prevents accidental letter triggering

## How it works

### Pipeline

```
collect_data.py  →  train.py  →  recognize.py
  (collect)           (train)      (deploy)
```

### Feature extraction (`collect_data.py`)
- MediaPipe detects 21 hand landmarks per frame (x, y, z coordinates → 63 values)
- Coordinates are normalised relative to the wrist (landmark 0) to remove position dependence
- Scale is normalised by dividing by the distance to the middle finger MCP (landmark 9), making predictions invariant to hand distance from camera
- Each sample is saved as a row in `training_data.csv`: `[label, f0, f1, ..., f62]`

### Training (`train.py`)
- Loads `training_data.csv` and trains an SVM with RBF kernel (`C=10`, `probability=True`)
- 80/20 train/test split; prints accuracy and full classification report
- Saves trained model to `model.pkl`

### Recognition (`recognize.py`)
- Loads `model.pkl` and runs inference on each frame
- Only accepts predictions with confidence > 60%
- Letter is added to the current word after being held for 0.8 seconds
- `SPACE` → speaks the current word via ElevenLabs TTS (queued, non-blocking)
- `BACKSPACE` → deletes last letter
- `Q` → quit

## Setup

```bash
pip install opencv-python mediapipe scikit-learn pandas pygame requests
```

Set your ElevenLabs API key as an environment variable:
```bash
export ELEVENLABS_API_KEY=your_key_here
```

## Usage

1. **Collect training data:**
   ```bash
   python collect_data.py
   ```
   Follow on-screen prompts. Press `SPACE` to begin collecting each sign.

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Run recognition:**
   ```bash
   python recognize.py
   ```

## Files

| File | Description |
|------|-------------|
| `collect_data.py` | Webcam-based data collection, saves to `training_data.csv` |
| `train.py` | Trains SVM classifier, saves to `model.pkl` |
| `recognize.py` | Real-time recognition + TTS output |
| `training_data.csv` | Collected landmark data (not included — run collect_data.py) |
| `model.pkl` | Trained model (not included — run train.py) |
| `hand_landmarker.task` | MediaPipe model (downloaded automatically) |

## Notes

- `hand_landmarker.task` is downloaded automatically on first run
- Works best in consistent lighting with a plain background
- Currently trained on letters I–Z; extend by editing `SIGNS` in `collect_data.py` and recollecting
