import json
import statistics
import time
from pathlib import Path

import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "R3iwan/kazakh-sentiment-bert"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models_kz"
TEXTS = [
    "Қызмет көрсету сапасы маған өте ұнады",
    "Фильм қызықты әрі әсерлі болды",
    "Нәтиже күткенімнен төмен болды",
    "Қызмет көрсету өте нашар, көңілім қалды",
    "Бұл тәжірибе жай ғана қалыпты өтті",
]


def summary(runtime, times, model_size_mb):
    ordered = sorted(times)
    return {
        "runtime": runtime,
        "model_size_mb": round(model_size_mb, 2),
        "mean_ms": round(statistics.mean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int((len(ordered) - 1) * 0.95)], 3),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")
    encoded = [tokenizer(text, return_tensors="np") for text in TEXTS]
    repeats = 30

    torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    torch_inputs = [tokenizer(text, return_tensors="pt") for text in TEXTS]
    for item in torch_inputs:
        torch_model(**item)
    torch_times = []
    for index in range(repeats):
        started = time.perf_counter()
        torch_model(**torch_inputs[index % len(torch_inputs)])
        torch_times.append((time.perf_counter() - started) * 1000)

    parameter_size = sum(p.nelement() * p.element_size() for p in torch_model.parameters()) / 1024 / 1024
    results = [summary("PyTorch FP32", torch_times, parameter_size)]

    for label, filename in [
        ("ONNX FP32", "kazakh_sentiment_fp32.onnx"),
        ("ONNX QUInt8", "kazakh_sentiment_uint8.onnx"),
    ]:
        path = MODEL_DIR / filename
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for item in encoded:
            session.run(None, dict(item))
        times = []
        for index in range(repeats):
            started = time.perf_counter()
            session.run(None, dict(encoded[index % len(encoded)]))
            times.append((time.perf_counter() - started) * 1000)
        results.append(summary(label, times, path.stat().st_size / 1024 / 1024))

    output = ROOT / "runtime_results_kz.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
