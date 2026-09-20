FROM python:3.10-slim

WORKDIR /app

# System deps needed by Pillow/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/main.py .
COPY saved_model/ ./saved_model/

ENV MODEL_DIR=saved_model
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
