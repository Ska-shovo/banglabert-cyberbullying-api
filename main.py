
import os
import json
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(BASE_DIR, "model")
)

app = FastAPI(
    title="Bangla Cyberbullying Detection API",
    description="Bangla/Banglish cyberbullying and online harassment detection using BanglaBERT.",
    version="1.0.0"
)

# Allow website frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading BanglaBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

# CPU inference for deployment
device = torch.device("cpu")
model.to(device)

# ------------------------------------------------------------
# Load label mapping
# ------------------------------------------------------------

with open(
    os.path.join(MODEL_DIR, "label_mapping.json"),
    "r",
    encoding="utf-8"
) as f:
    label_mapping = json.load(f)

id2label = {
    int(k): v
    for k, v in label_mapping.items()
}

print("Model loaded successfully.")

# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class TextRequest(BaseModel):
    text: str

# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "BanglaBERT",
        "classes": list(id2label.values())
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

@app.post("/predict")
def predict(request: TextRequest):

    text = request.text.strip()

    if not text:
        return {
            "error": "Text cannot be empty."
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_id].item()
    )

    all_probabilities = {
        id2label[i]: round(
            float(probabilities[i].item()),
            6
        )
        for i in range(len(probabilities))
    }

    return {
        "text": text,
        "prediction": id2label[predicted_id],
        "confidence": round(confidence, 6),
        "probabilities": all_probabilities
    }
