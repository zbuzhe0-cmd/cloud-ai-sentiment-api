# Cloud AI Sentiment API

Облачный API для анализа тональности русскоязычных отзывов.

- Модель: `seara/rubert-tiny2-russian-sentiment`
- Архитектура: RuBERT Tiny 2
- Классы: `positive`, `negative`, `neutral`
- API: FastAPI
- Развёртывание: Docker / Render

## API

Проверка состояния:

```http
GET /health
```

Анализ текста:

```http
POST /predict
Content-Type: application/json

{"text":"Мне очень понравился сервис"}
```

## Локальный запуск

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

## Render

Репозиторий содержит `render.yaml` и `Dockerfile`. Для бесплатного размещения создайте Render Blueprint из этого репозитория и выберите план Free.

Бесплатный сервис засыпает после периода без запросов, поэтому первый запрос после простоя может выполняться около минуты.

## Исследовательский контур

В `research/` находятся воспроизводимые скрипты:

- `export_onnx.py` — экспорт PyTorch в ONNX FP32 и динамическая INT8-квантизация;
- `compare_runtimes.py` — 30 прогонов каждого runtime с расчётом mean, P50 и P95;
- `runtime_results.json` — зафиксированный результат контрольного эксперимента.

Контрольный прогон показал уменьшение размера модели с 117 МБ до 28,03 МБ и ускорение среднего инференса примерно в 3,85 раза для ONNX INT8 относительно PyTorch FP32 на локальном CPU. Результаты зависят от оборудования и должны повторяться в целевой среде.
