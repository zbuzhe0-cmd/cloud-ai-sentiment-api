from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
print(f"Downloaded {MODEL_NAME}")

