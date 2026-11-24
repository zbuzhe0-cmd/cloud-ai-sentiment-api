import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
ROOT = Path(__file__).resolve().parent
TEXTS = [
    "Сервис работает быстро, результатом полностью доволен",
    "Обычная услуга, ничего особенного не произошло",
    "Ужасное обслуживание, больше никогда не обращусь",
]


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def summarize(name, values, size_mb):
    return {
        "runtime": name,
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "model_size_mb": round(size_mb, 2),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = [tokenizer(t, return_tensors="np") for t in TEXTS]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    torch_inputs = [tokenizer(t, return_tensors="pt") for t in TEXTS]

    with torch.inference_mode():
        for item in torch_inputs:
            model(**item)
        torch_times = []
        for i in range(30):
            started = time.perf_counter()
            model(**torch_inputs[i % len(torch_inputs)])
            torch_times.append((time.perf_counter() - started) * 1000)

    results = [summarize("PyTorch FP32", torch_times, 117.0)]
    for label, filename in [("ONNX FP32", "rubert_sentiment_fp32.onnx"), ("ONNX INT8", "rubert_sentiment_int8.onnx")]:
        path = ROOT / "models" / filename
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for item in encoded:
            session.run(None, {"input_ids": item["input_ids"].astype(np.int64), "attention_mask": item["attention_mask"].astype(np.int64)})
        times = []
        for i in range(30):
            item = encoded[i % len(encoded)]
            started = time.perf_counter()
            session.run(None, {"input_ids": item["input_ids"].astype(np.int64), "attention_mask": item["attention_mask"].astype(np.int64)})
            times.append((time.perf_counter() - started) * 1000)
        results.append(summarize(label, times, path.stat().st_size / 1024 / 1024))

    output = ROOT / "runtime_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
