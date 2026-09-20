"""Live webcam emotion detection.

Opens your laptop's default camera, detects faces frame-by-frame using
OpenCV's built-in Haar cascade, runs each detected face through the trained
model, and overlays the predicted emotion + confidence directly on the
video feed in a window. Press 'q' to quit.

This runs standalone (loads the model directly) -- it does NOT need the
FastAPI server or Streamlit running.

Usage:
    python model/webcam_demo.py --model saved_model/emotion_model.h5 --labels saved_model/labels.json

Requires the regular (non-headless) opencv-python package, since headless
builds don't support cv2.imshow():
    pip install opencv-python
"""
import argparse
import json
import time

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 215, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="saved_model/emotion_model.h5")
    p.add_argument("--labels", default="saved_model/labels.json")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading model...")
    model = load_model(args.model)
    with open(args.labels) as f:
        meta = json.load(f)
    labels = meta["labels"]
    img_size = meta.get("img_size", 96)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try a different --camera value."
        )

    print("Camera open. Press 'q' in the video window to quit.")
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (img_size, img_size))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype("float32")
            face_input = preprocess_input(face_rgb)
            face_input = np.expand_dims(face_input, axis=0)

            probs = model.predict(face_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            emotion = labels[pred_idx]
            confidence = probs[pred_idx]

            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{emotion} ({confidence:.0%})"
            cv2.putText(
                frame, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        # FPS counter, helpful to see if it's keeping up on your laptop
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        cv2.imshow("Live Emotion Recognition - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
