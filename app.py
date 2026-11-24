import os
from time import perf_counter

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = os.getenv("MODEL_NAME", "seara/rubert-tiny2-russian-sentiment")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

app = FastAPI(title="Cloud AI Sentiment API", version="2.0.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def normalize_label(label: str) -> str:
    normalized = label.lower().replace("_", " ")
    if "negative" in normalized:
        return "negative"
    if "positive" in normalized:
        return "positive"
    return "neutral"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "framework": "PyTorch + Transformers",
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    started = perf_counter()
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    best_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[best_index].item())
    raw_label = str(model.config.id2label[best_index])
    elapsed_ms = (perf_counter() - started) * 1000

    return {
        "sentiment": normalize_label(raw_label),
        "raw_label": raw_label,
        "confidence": round(confidence, 4),
        "inference_ms": round(elapsed_ms, 2),
        "model": MODEL_NAME,
    }

