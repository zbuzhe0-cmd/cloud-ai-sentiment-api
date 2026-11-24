import os
from time import perf_counter

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer


MODEL_NAME = "R3iwan/kazakh-sentiment-bert"
MODEL_PATH = os.getenv("MODEL_PATH", "/app/kazakh_sentiment_uint8.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/app/tokenizer")
LABELS = ["negative", "neutral", "positive"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions(),
)

app = FastAPI(
    title="Қазақ тіліндегі тоналдылықты талдау API",
    description="ONNX QUInt8 форматындағы қазақша BERT моделі",
    version="3.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000, description="Қазақша мәтін")


@app.get("/")
def root():
    return {
        "service": "Қазақ тіліндегі тоналдылықты талдау",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "framework": "ONNX Runtime QUInt8",
        "language": "kk",
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if len(payload.text.split()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Модельге контекст қажет. Кемінде үш сөзден тұратын толық пікір енгізіңіз.",
        )
    started = perf_counter()
    encoded = tokenizer(
        payload.text,
        return_tensors="np",
        truncation=True,
        max_length=256,
    )
    feeds = {name: value.astype(np.int64) for name, value in encoded.items()}
    logits = session.run(None, feeds)[0][0]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    best_index = int(np.argmax(probabilities))
    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "sentiment": LABELS[best_index],
        "raw_label": LABELS[best_index],
        "confidence": round(float(probabilities[best_index]), 4),
        "inference_ms": round(elapsed_ms, 2),
        "model": MODEL_NAME,
        "runtime": "ONNX Runtime QUInt8",
    }

