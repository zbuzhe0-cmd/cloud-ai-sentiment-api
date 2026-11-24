from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "R3iwan/kazakh-sentiment-bert"
OUTPUT_DIR = Path(__file__).resolve().parent / "models_kz"
FP32_PATH = OUTPUT_DIR / "kazakh_sentiment_fp32.onnx"
INT8_PATH = OUTPUT_DIR / "kazakh_sentiment_int8.onnx"


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    sample = tokenizer("Бұл қызмет маған өте ұнады", return_tensors="pt")
    wrapper = LogitsOnly(model)
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        FP32_PATH,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    print(f"FP32: {FP32_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"INT8: {INT8_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
