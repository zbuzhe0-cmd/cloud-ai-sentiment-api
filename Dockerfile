FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_NAME=seara/rubert-tiny2-russian-sentiment \
    HF_HOME=/app/model_cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir typing-extensions filelock sympy networkx jinja2 fsspec && \
    pip install --no-cache-dir --no-deps \
      --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY download_model.py .
RUN python download_model.py

COPY app.py .

CMD exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1

