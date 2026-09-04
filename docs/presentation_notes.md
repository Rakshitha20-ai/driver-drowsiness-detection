# 3-Hour Presentation Notes

## 1. Problem
Detect possible driver drowsiness from eye closure in a webcam stream.

## 2. Architecture
Webcam → Frame → Grayscale → Haar Cascade → Eye Crop → Resize → Normalize → CNN → Open/Closed → Temporal Logic → Alert.

## 3. OpenCV
Captures frames, converts images, detects face/eyes, draws the UI and displays the result.

## 4. Haar Cascade
A classical detector used here to locate the face and eyes. It proposes regions; the CNN classifies the eye state.

## 5. CNN
The CNN learns visual features from labeled open/closed eye images.

Reference architecture:
24×24×1 → Conv2D(32) → Conv2D(32) → Conv2D(64) → Dropout → Flatten → Dense(128) → Dropout → Softmax(2).

## 6. Preprocessing
Grayscale → 24×24 resize → pixel values /255 → batch dimension.

## 7. Why temporal logic?
One closed-eye frame is not enough to call a driver drowsy. Sustained closure is a stronger signal and reduces false alarms.

## 8. Improvements
- confidence threshold
- consecutive-frame/time threshold
- alarm cooldown
- FPS/latency measurement
- confusion matrix and precision/recall/F1
- improved CNN with normal pooling
- better dataset and robustness tests

## 9. Interview statement
“I built a real-time driver drowsiness detection prototype using OpenCV and a CNN. OpenCV captures frames and detects the face and eyes with Haar Cascades. Eye crops are converted to grayscale, resized to 24×24 and normalized. The CNN classifies eyes as open or closed, and temporal logic turns sustained closure into a drowsiness alert. I then focused on improving robustness, evaluation and reproducibility rather than treating a single-frame prediction as drowsiness.”
