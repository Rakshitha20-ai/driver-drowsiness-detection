import cv2
import numpy as np

def preprocess_eye(eye_bgr, image_size=(24, 24)):
    """Convert an eye crop into the CNN input format: 1 x H x W x 1."""
    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
