from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
OUTPUT_DIR = Path(__file__).resolve().parent / "models"
FP32_PATH = OUTPUT_DIR / "rubert_sentiment_fp32.onnx"
INT8_PATH = OUTPUT_DIR / "rubert_sentiment_int8.onnx"


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    wrapper = LogitsOnly(model)
    sample = tokenizer("Тестовая последовательность", return_tensors="pt")

    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        FP32_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    print(f"FP32: {FP32_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"INT8: {INT8_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

