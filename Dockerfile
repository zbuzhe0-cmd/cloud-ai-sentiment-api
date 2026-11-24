FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/kazakh_sentiment_uint8.onnx \
    TOKENIZER_PATH=/app/tokenizer \
    HF_HOME=/app/model_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY download_tokenizer.py .
RUN python download_tokenizer.py && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/zbuzhe0-cmd/cloud-ai-sentiment-api/releases/download/v3-kazakh/kazakh_sentiment_uint8.onnx', '/app/kazakh_sentiment_uint8.onnx')"

COPY app.py .

CMD exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1

