# Google Cloud Run image. scripts/deploy_cloud_run.py stages the app, the
# checkpoint and metrics.json next to a copy of this file and builds it with
# Cloud Build, so the model is baked in and nothing is fetched at boot.
#   - tensorflow-cpu: same 2.13 API, no CUDA payload (Cloud Run has no GPU here).
#   - PORT is set by Cloud Run (8080); app.py honours it.
#   - PNEUMOSCAN_PUBLIC=1: the audit log is world-readable on a public URL,
#     so it shows only a prefix of each case uid.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN sed -e 's/^tensorflow==/tensorflow-cpu==/' -e '/^pytest==/d' requirements.txt > req-run.txt \
    && pip install --no-cache-dir -r req-run.txt

COPY . .

ENV PNEUMOSCAN_PUBLIC=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

CMD ["python", "app/app.py"]
