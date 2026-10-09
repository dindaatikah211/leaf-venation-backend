FROM python:3.12-slim

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

ENV USE_TF=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY leaf_guard.py .
RUN python -c "import leaf_guard"

COPY . .

CMD gunicorn -b 0.0.0.0:${PORT:-8000} --workers 1 --threads 2 --timeout 120 app:app