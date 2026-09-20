"""FastAPI REST API for real-time facial emotion prediction.

Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> upload an image, get back the predicted emotion and
                       per-class confidence scores
"""
import io
import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_DIR = os.environ.get("MODEL_DIR", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Upload a face image and get back predicted emotion + confidence scores.",
    version="1.0.0",
)

# Allow the Streamlit frontend (any origin, since the deployed URL varies)
# to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_labels = None
_img_size = 96


def detect_and_crop_face(pil_image: Image.Image):
    """Detects the largest face in the image and crops to it, with a small
    margin. Falls back to the full image if no face is found -- this keeps
    prediction quality high for uploads that aren't already tightly cropped
    to a face (e.g. a normal phone photo with background), matching how the
    model was trained on close-up face crops. Returns (image, face_found)."""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return pil_image, False

    # Pick the largest detected face (by area) in case of multiple faces.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add ~15% margin around the face so we don't crop too tightly.
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    x0 = max(x - margin_x, 0)
    y0 = max(y - margin_y, 0)
    x1 = min(x + w + margin_x, cv_image.shape[1])
    y1 = min(y + h + margin_y, cv_image.shape[0])

    face_crop = cv_image[y0:y1, x0:x1]
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_crop_rgb), True


def get_model():
    """Lazily loads the model on first request so the container starts fast
    and doesn't fail hard if the model file is briefly unavailable."""
    global _model, _labels, _img_size
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found at {MODEL_PATH}. Train the model first.",
            )
        _model = load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            meta = json.load(f)
        _labels = meta["labels"]
        _img_size = meta.get("img_size", 96)
    return _model, _labels, _img_size


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    model, labels, img_size = get_model()

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    image, face_found = detect_and_crop_face(image)
    image = image.resize((img_size, img_size))
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_emotion": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {label: float(p) for label, p in zip(labels, probs)},
        "face_detected": face_found,
    }