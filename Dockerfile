FROM python:3.12-slim

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn -b 0.0.0.0:${PORT:-8000} --workers 1 --threads 2 --timeout 120 app:app