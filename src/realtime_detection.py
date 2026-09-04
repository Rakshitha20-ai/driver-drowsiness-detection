import time
from pathlib import Path

import cv2
import numpy as np

from config import (
    FACE_CASCADE,
    LEFT_EYE_CASCADE,
    RIGHT_EYE_CASCADE,
    MODEL_PATH,
    ALARM_PATH,
    IMAGE_SIZE,
    CLOSED_FRAME_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)

# Optional audio dependency. The system can still display the alert if audio fails.
try:
    from pygame import mixer
except ImportError:
    mixer = None

try:
    from keras.models import load_model
except ImportError as exc:
    raise ImportError("Install the project requirements inside the virtual environment.") from exc

from preprocessing import preprocess_eye


def load_assets():
    missing = [str(p) for p in [FACE_CASCADE, LEFT_EYE_CASCADE, RIGHT_EYE_CASCADE, MODEL_PATH]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing project assets:\n" + "\n".join(missing)
            + "\nPlace the required cascade/model files in the project folders."
        )

    face = cv2.CascadeClassifier(str(FACE_CASCADE))
    left_eye = cv2.CascadeClassifier(str(LEFT_EYE_CASCADE))
    right_eye = cv2.CascadeClassifier(str(RIGHT_EYE_CASCADE))
    model = load_model(str(MODEL_PATH))
    return face, left_eye, right_eye, model


def predict_eye(model, eye):
    x = preprocess_eye(eye, IMAGE_SIZE)
    probabilities = model.predict(x, verbose=0)[0]
    label = int(np.argmax(probabilities))
    confidence = float(probabilities[label])
    return label, confidence


def main():
    face, left_eye_detector, right_eye_detector, model = load_assets()

    sound = None
    if mixer is not None and ALARM_PATH.exists():
        mixer.init()
        sound = mixer.Sound(str(ALARM_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    closed_frames = 0
    last_alarm_time = 0.0
    alarm_cooldown = 3.0

    previous_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
            )

            eye_states = []

            for (x, y, w, h) in faces[:1]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
                face_gray = gray[y:y+h, x:x+w]

                left = left_eye_detector.detectMultiScale(face_gray)
                right = right_eye_detector.detectMultiScale(face_gray)

                for ex, ey, ew, eh in list(left[:1]) + list(right[:1]):
                    eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                    label, confidence = predict_eye(model, eye)

                    if confidence >= CONFIDENCE_THRESHOLD:
                        eye_states.append(label)

            # Count a frame as closed only when both detected eyes are confidently closed.
            if len(eye_states) >= 2 and all(s == 0 for s in eye_states[:2]):
                closed_frames += 1
            elif len(eye_states) > 0:
                closed_frames = max(0, closed_frames - 1)

            status = "DROWSY" if closed_frames >= CLOSED_FRAME_THRESHOLD else "ALERT"

            if status == "DROWSY":
                cv2.rectangle(
                    frame, (0, 0),
                    (frame.shape[1] - 1, frame.shape[0] - 1),
                    (0, 0, 255), 4
                )

                now = time.perf_counter()
                if sound is not None and now - last_alarm_time >= alarm_cooldown:
                    sound.play()
                    last_alarm_time = now

            cv2.putText(
                frame, f"Status: {status}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Closed frames: {closed_frames}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
