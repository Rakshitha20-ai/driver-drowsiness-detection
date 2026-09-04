import time

import cv2
import numpy as np

from config import (
    FACE_CASCADE,
    LEFT_EYE_CASCADE,
    RIGHT_EYE_CASCADE,
    MODEL_PATH,
    ALARM_PATH,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
)

try:
    from pygame import mixer
except ImportError:
    mixer = None

try:
    from keras.models import load_model
except ImportError as exc:
    raise ImportError(
        "Install the project requirements inside the virtual environment."
    ) from exc

from preprocessing import preprocess_eye


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

# Eyes must remain closed for this many seconds
# before the system declares the driver drowsy.
DROWSY_TIME_THRESHOLD = 2.0

# Alarm can play again only after this many seconds.
ALARM_COOLDOWN = 3.0


# --------------------------------------------------
# LOAD MODEL AND DETECTORS
# --------------------------------------------------

def load_assets():
    missing = [
        str(p)
        for p in [
            FACE_CASCADE,
            LEFT_EYE_CASCADE,
            RIGHT_EYE_CASCADE,
            MODEL_PATH,
        ]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing project assets:\n"
            + "\n".join(missing)
            + "\nPlace the required files in the project folders."
        )

    face = cv2.CascadeClassifier(str(FACE_CASCADE))
    left_eye = cv2.CascadeClassifier(str(LEFT_EYE_CASCADE))
    right_eye = cv2.CascadeClassifier(str(RIGHT_EYE_CASCADE))

    model = load_model(str(MODEL_PATH))

    return face, left_eye, right_eye, model


# --------------------------------------------------
# PREDICT EYE STATE
# --------------------------------------------------

def predict_eye(model, eye):
    """
    Predict whether an eye is closed or open.

    Returns:
        label      -> 0 = Closed, 1 = Open
        confidence -> model confidence
    """

    x = preprocess_eye(eye, IMAGE_SIZE)

    probabilities = model.predict(
        x,
        verbose=0
    )[0]

    label = int(np.argmax(probabilities))

    confidence = float(
        probabilities[label]
    )

    return label, confidence


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    face, left_eye_detector, right_eye_detector, model = load_assets()

    # -----------------------------
    # Initialize alarm
    # -----------------------------

    sound = None

    if mixer is not None and ALARM_PATH.exists():

        mixer.init()

        sound = mixer.Sound(
            str(ALARM_PATH)
        )

    # -----------------------------
    # Open webcam
    # -----------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open the webcam."
        )

    # -----------------------------
    # Tracking variables
    # -----------------------------

    eyes_closed_start = None

    last_alarm_time = 0.0

    previous_time = time.perf_counter()

    try:

        while True:

            # --------------------------------
            # Read webcam frame
            # --------------------------------

            ret, frame = cap.read()

            if not ret:
                break

            height, width = frame.shape[:2]

            # --------------------------------
            # Convert frame to grayscale
            # --------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # --------------------------------
            # Detect face
            # --------------------------------

            faces = face.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(25, 25)
            )

            eye_states = []
            eye_confidences = []

            # --------------------------------
            # Process first detected face
            # --------------------------------

            for (x, y, w, h) in faces[:1]:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (100, 100, 100),
                    1
                )

                face_gray = gray[
                    y:y + h,
                    x:x + w
                ]

                # Detect left and right eyes
                left = left_eye_detector.detectMultiScale(
                    face_gray
                )

                right = right_eye_detector.detectMultiScale(
                    face_gray
                )

                detected_eyes = (
                    list(left[:1])
                    + list(right[:1])
                )

                # --------------------------------
                # Predict each detected eye
                # --------------------------------

                for ex, ey, ew, eh in detected_eyes:

                    eye = frame[
                        y + ey:y + ey + eh,
                        x + ex:x + ex + ew
                    ]

                    label, confidence = predict_eye(
                        model,
                        eye
                    )

                    # Only trust predictions
                    # above confidence threshold.
                    if confidence >= CONFIDENCE_THRESHOLD:

                        eye_states.append(label)

                        eye_confidences.append(
                            confidence
                        )

            # --------------------------------
            # Determine eye condition
            # --------------------------------

            both_eyes_closed = (
                len(eye_states) >= 2
                and all(
                    state == 0
                    for state in eye_states[:2]
                )
            )

            # --------------------------------
            # Track how long eyes stay closed
            # --------------------------------

            if both_eyes_closed:

                if eyes_closed_start is None:

                    eyes_closed_start = time.perf_counter()

                closed_duration = (
                    time.perf_counter()
                    - eyes_closed_start
                )

            else:

                eyes_closed_start = None

                closed_duration = 0.0

            # --------------------------------
            # Determine system status
            # --------------------------------

            if closed_duration >= 2.0:
                status = "CRITICAL"
                
            elif closed_duration >= 1.0:
                status = "DROWSY"

            elif closed_duration > 0:
                status = "WARNING"

            else:
                status = "ALERT"

            # --------------------------------
            # Alarm
            # --------------------------------

            if status in ("DROWSY", "CRITICAL"):

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (width - 1, height - 1),
                    (0, 0, 255),
                    4
                )

                current_time = time.perf_counter()

                if (
                    sound is not None
                    and current_time - last_alarm_time
                    >= ALARM_COOLDOWN
                ):

                    sound.play()

                    last_alarm_time = current_time

            # --------------------------------
            # Calculate confidence
            # --------------------------------

            if eye_confidences:

                average_confidence = (
                    sum(eye_confidences)
                    / len(eye_confidences)
                )

            else:

                average_confidence = 0.0

            # --------------------------------
            # Calculate FPS
            # --------------------------------

            current_time = time.perf_counter()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6
            )

            previous_time = current_time

            # --------------------------------
            # Display information
            # --------------------------------

            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Closed duration: {closed_duration:.1f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {average_confidence:.2f}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # --------------------------------
            # Show webcam
            # --------------------------------

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):

                break

    finally:

        cap.release()

        cv2.destroyAllWindows()


# --------------------------------------------------
# START PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()