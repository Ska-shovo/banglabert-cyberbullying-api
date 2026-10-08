# Deployment Structure

This repository contains the FastAPI backend only.

The BanglaBERT model is intentionally NOT included in GitHub
because the model package is approximately 423 MB.

Current model:
csebuetnlp/banglabert

Classes:
1. Harassment
2. Hate Speech
3. Neutral
4. Sexual Aggression
5. Violent Extremism

API endpoint:
POST /predict

The trained model remains safely backed up in Google Drive.
