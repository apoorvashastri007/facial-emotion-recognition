# Facial Emotion Recognition System

End-to-end deep learning application that classifies facial expressions into
**7 emotions** — Happy, Sad, Angry, Fear, Surprise, Neutral, Disgust — using a
CNN built with transfer learning on **MobileNetV2**. Served through a
**FastAPI** REST backend and an **interactive Streamlit** frontend, both
containerized with **Docker / Docker Compose**.

```
facial-emotion-recognition/
├── data/                       # dataset goes here (see below)
├── model/
│   ├── preprocess.py           # data loading + augmentation
│   ├── model.py                # MobileNetV2 transfer-learning architecture
│   ├── train.py                # training script (2-phase: head + fine-tune)
│   └── evaluate.py             # confusion matrix / classification report
├── api/
│   ├── main.py                 # FastAPI app (POST /predict, GET /health)
│   ├── requirements.txt
│   └── Dockerfile
├── app/
│   ├── streamlit_app.py        # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── saved_model/                # trained .h5 model + labels.json land here
├── docker-compose.yml
├── requirements.txt             # training environment deps
└── .gitignore
```

## 1. Dataset

This project is built around the standard **FER2013** dataset (or any dataset
with the same folder layout), organized as:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── ... (same 7 classes)
```

Download it from Kaggle, e.g. "FER-2013" or "Face expression recognition
dataset", and unzip it into `data/` so it matches the layout above. Grayscale
48x48 images are fine — `preprocess.py` resizes and converts to RGB
automatically for MobileNetV2.

## 2. Train the model

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python model/train.py \
  --data-dir data \
  --img-size 96 \
  --batch-size 64 \
  --epochs-head 15 \
  --epochs-finetune 15
```

This runs two phases:
1. **Head training** — MobileNetV2 base frozen, only the new classification
   head trains (fast, stabilizes the new layers).
2. **Fine-tuning** — top ~30 layers of MobileNetV2 unfrozen and trained at a
   low learning rate to adapt pretrained ImageNet features to faces.

Outputs land in `saved_model/`:
- `emotion_model.h5` — the trained Keras model
- `labels.json` — index-to-emotion-label mapping
- `training_history.png` — accuracy/loss curves

Check `model/evaluate.py` for a confusion matrix + classification report on
the test split once training finishes.

## 3. Run the API locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test it:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@/path/to/face.jpg"
```

Response:
```json
{
  "predicted_emotion": "happy",
  "confidence": 0.87,
  "probabilities": {
    "angry": 0.01, "disgust": 0.00, "fear": 0.02, "happy": 0.87,
    "neutral": 0.05, "sad": 0.02, "surprise": 0.03
  }
}
```

## 4. Run the Streamlit app locally

```bash
cd app
pip install -r requirements.txt
export API_URL=http://localhost:8000
streamlit run streamlit_app.py
```

## 5. Run everything with Docker Compose

Make sure `saved_model/emotion_model.h5` and `saved_model/labels.json` exist
(from step 2) before building — the API Dockerfile copies them in.

```bash
docker compose up --build
```

- API → http://localhost:8000/docs (Swagger UI)
- Streamlit UI → http://localhost:8501

## 6. Deployment

### Backend (FastAPI) → Render
1. Push this repo to GitHub.
2. On Render: **New → Web Service**, connect the repo, root directory `api/`.
3. Environment: Docker (Render will pick up `api/Dockerfile`).
4. Set instance type, deploy. Note the public URL, e.g.
   `https://your-api.onrender.com`.

### Frontend (Streamlit) → Streamlit Community Cloud
1. On https://share.streamlit.io, **New app**, point to this repo,
   `app/streamlit_app.py` as the entry point.
2. In **Secrets**, add:
   ```
   API_URL = "https://your-api.onrender.com"
   ```
3. Deploy. The app reads `API_URL` from `st.secrets` (falls back to the
   `API_URL` env var, then localhost, for local dev).

## Tech Stack
Python, TensorFlow, Keras, OpenCV, MobileNetV2, FastAPI, Streamlit, Docker,
NumPy, Matplotlib.
