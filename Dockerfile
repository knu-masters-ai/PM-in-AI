# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY ui ./ui
RUN mkdir -p /app/.streamlit
COPY ui/.streamlit/config.toml /app/.streamlit/config.toml
COPY supervisord.conf ./supervisord.conf

EXPOSE 8000 8501

ENV BACKEND_URL=http://localhost:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["supervisord", "-c", "/app/supervisord.conf"]
