FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 libgl1 libglib2.0-0 supervisor \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY ui ./ui
RUN mkdir -p /app/.streamlit
COPY ui/.streamlit/config.toml /app/.streamlit/config.toml
COPY api/weights/best.onnx ./api/weights/best.onnx
COPY supervisord.conf ./supervisord.conf

EXPOSE 8000 8501
ENV MODEL_WEIGHTS=/app/api/weights/best.onnx \
    BACKEND_URL=http://localhost:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["supervisord", "-c", "/app/supervisord.conf"]
