from transformers import AutoTokenizer

MODEL_NAME = "R3iwan/kazakh-sentiment-bert"
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained("/app/tokenizer")
print(f"Tokenizer downloaded: {MODEL_NAME}")
