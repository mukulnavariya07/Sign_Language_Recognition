from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; MODELS=ROOT/'models'; REPORTS=ROOT/'reports'
CSV=DATA/'landmarks.csv'; MODEL=MODELS/'asl_model.joblib'
TASK=MODELS/'hand_landmarker.task'
TASK_URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
LABELS=list('ABCDEFGHIKLMNOPQRSTUVWXY') # J and Z are dynamic signs
N_FEATURES=63
