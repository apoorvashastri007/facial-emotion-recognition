"""Streamlit web app: upload a face image, call the FastAPI backend,
visualize the predicted emotion and confidence scores."""
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="centered")


def get_api_url() -> str:
    # Priority: Streamlit secrets (for Streamlit Community Cloud) ->
    # environment variable (for Docker Compose / local) -> localhost fallback.
    # We only touch st.secrets if a secrets.toml file actually exists --
    # accessing st.secrets at all triggers Streamlit's own "no secrets
    # file found" warning banner locally, even inside try/except.
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            return st.secrets["API_URL"]
        except Exception:
            pass
    return os.environ.get("API_URL", "http://localhost:8000")


API_URL = get_api_url()

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

st.title("🙂 Facial Emotion Recognition")
st.write(
    "Upload a photo containing a face and the model will classify the "
    "expression into one of seven emotions: Happy, Sad, Angry, Fear, "
    "Surprise, Neutral, Disgust."
)

with st.sidebar:
    st.subheader("Settings")
    st.text_input("API URL", value=API_URL, disabled=True, help="Set via secrets/env var")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        status = "🟢 online" if health.get("model_loaded") else "🟡 online, model not loaded"
    except Exception:
        status = "🔴 unreachable"
    st.write(f"**API status:** {status}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    with st.spinner("Predicting emotion..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the prediction API: {e}")
            result = None

    if result:
        emotion = result["predicted_emotion"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        if not result.get("face_detected", True):
            st.warning(
                "No face was clearly detected in this image — the model ran on "
                "the full image, which can reduce accuracy. Try a clearer, "
                "front-facing photo where the face is well-lit and unobstructed."
            )

        with col2:
            emoji = EMOTION_EMOJI.get(emotion, "")
            st.metric("Predicted Emotion", f"{emoji} {emotion.capitalize()}", f"{confidence:.1%} confidence")

        st.subheader("Confidence scores")
        df = pd.DataFrame(
            {"emotion": list(probs.keys()), "confidence": list(probs.values())}
        ).sort_values("confidence", ascending=False)
        st.bar_chart(df.set_index("emotion"))
        st.dataframe(df.style.format({"confidence": "{:.2%}"}))
else:
    st.info("Upload an image to get started.")