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
POSITIVE_WORDS = {
    "жақсы", "керемет", "тамаша", "ұнады", "сапалы", "ризамын", "қуаныштымын",
}
NEGATIVE_WORDS = {
    "жаман", "нашар", "ұнамады", "сапасыз", "көңілім қалды", "өкінішті",
}

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
    model_sentiment = LABELS[best_index]
    normalized = payload.text.lower().replace("ё", "е")
    positive_hits = sum(phrase in normalized for phrase in POSITIVE_WORDS)
    negative_hits = sum(phrase in normalized for phrase in NEGATIVE_WORDS)
    calibrated_sentiment = model_sentiment
    decision_source = "bert_model"
    confidence = float(probabilities[best_index])
    if positive_hits > negative_hits:
        calibrated_sentiment = "positive"
        decision_source = "lexical_calibration"
        confidence = max(0.90, min(0.99, 0.90 + 0.03 * positive_hits))
    elif negative_hits > positive_hits:
        calibrated_sentiment = "negative"
        decision_source = "lexical_calibration"
        confidence = max(0.90, min(0.99, 0.90 + 0.03 * negative_hits))
    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "sentiment": calibrated_sentiment,
        "raw_label": model_sentiment,
        "decision_source": decision_source,
        "confidence": round(confidence, 4),
        "inference_ms": round(elapsed_ms, 2),
        "model": MODEL_NAME,
        "runtime": "ONNX Runtime QUInt8",
    }

