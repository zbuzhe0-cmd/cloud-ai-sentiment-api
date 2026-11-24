FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_NAME=seara/rubert-tiny2-russian-sentiment \
    HF_HOME=/app/model_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY app.py .

CMD exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1

