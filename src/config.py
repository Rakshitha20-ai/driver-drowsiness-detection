from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACE_CASCADE = PROJECT_ROOT / "haar cascade files" / "haarcascade_frontalface_alt.xml"
LEFT_EYE_CASCADE = PROJECT_ROOT / "haar cascade files" / "haarcascade_lefteye_2splits.xml"
RIGHT_EYE_CASCADE = PROJECT_ROOT / "haar cascade files" / "haarcascade_righteye_2splits.xml"

MODEL_PATH = PROJECT_ROOT / "models" / "cnnCat2.h5"
ALARM_PATH = PROJECT_ROOT / "alarm.wav"

IMAGE_SIZE = (24, 24)
CLOSED_LABEL = 0
OPEN_LABEL = 1

# First version: deliberately conservative. We will tune this experimentally.
CLOSED_FRAME_THRESHOLD = 15
CONFIDENCE_THRESHOLD = 0.60
