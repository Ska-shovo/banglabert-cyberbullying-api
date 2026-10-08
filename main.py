
import json
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face model repository
MODEL_ID = "kumarshuvo605/banglabert-cyberbullying"

app = FastAPI(
    title="Bangla Cyberbullying Detection API",
    description="Bangla/Banglish cyberbullying and online harassment detection using fine-tuned BanglaBERT.",
    version="1.0.0"
)

# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Load model from Hugging Face
# ------------------------------------------------------------

print(f"Loading model from Hugging Face: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID
)

device = torch.device("cpu")
model.to(device)
model.eval()

print("Model loaded successfully.")

# ------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------

id2label = {
    0: "Harassment",
    1: "Hate Speech",
    2: "Neutral",
    3: "Sexual Aggression",
    4: "Violent Extremism"
}

# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class TextRequest(BaseModel):
    text: str

# ------------------------------------------------------------
# Root endpoint
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "model": MODEL_ID,
        "classes": list(id2label.values())
    }

# ------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_ID
    }

# ------------------------------------------------------------
# Prediction endpoint
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
