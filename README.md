# Real-Time Driver Drowsiness Detection

A computer-vision prototype that detects eye state from a webcam and uses temporal logic to identify possible drowsiness.

## Project goal

The system processes webcam frames, detects the driver's face and eyes with OpenCV Haar Cascades, classifies each eye as **Open** or **Closed** using a CNN, and triggers an alert when closed-eye evidence persists.

> This is an educational/safety prototype, not a medical or automotive-certified system.

## Pipeline

Webcam → Frame → Grayscale → Face/Eye Detection → Eye Crop → 24×24 Resize → Pixel Normalization → CNN → Open/Closed → Temporal Drowsiness Score → Alert

## Current baseline

The working reference baseline uses:
- OpenCV for webcam and Haar Cascade detection
- 24×24 grayscale eye images
- CNN binary classification (Open/Closed)
- a running closed-eye score
- an alarm when the score crosses a threshold

## Planned original improvements

1. Replace the simple frame counter with time/consecutive-frame logic.
2. Add prediction confidence thresholds.
3. Add an alarm cooldown to avoid repeated triggering.
4. Measure FPS and inference latency.
5. Evaluate precision, recall, F1-score and confusion matrix.
6. Compare the reference CNN with an improved CNN using normal 2×2 pooling.
7. Build a reproducible data-capture and training pipeline.
8. Improve robustness to glasses, lighting and missed detections.

## Repository structure

```text
driver-drowsiness-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   └── realtime_detection.py
├── docs/
│   └── presentation_notes.md
├── data/          # keep large/private datasets out of Git
└── models/        # keep large model binaries out of Git
```

## Reference

This project was developed by studying an existing open-source drowsiness-detection implementation and then planning a clean, modular reimplementation with measurable improvements. The reference should be credited; copied code/model files should not be represented as original work.

## Status

- [x] Working webcam baseline
- [x] Environment setup
- [x] CNN model inspection
- [ ] Original dataset pipeline
- [ ] Improved CNN
- [ ] Quantitative evaluation
- [ ] Final demo
