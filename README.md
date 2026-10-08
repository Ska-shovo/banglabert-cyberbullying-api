# BanglaBERT Cyberbullying Detection API

FastAPI backend for Bangla/Banglish cyberbullying and online harassment detection.

## Model

CSEBUET NLP BanglaBERT fine-tuned on the BenCyber dataset.

## Classes

- Harassment
- Hate Speech
- Neutral
- Sexual Aggression
- Violent Extremism

## API

### GET /
API status.

### GET /health
Health check.

### POST /predict

Request:

{
  "text": "তোমার কথাটা খুব সুন্দর"
}

Returns the predicted class, confidence and probabilities.
