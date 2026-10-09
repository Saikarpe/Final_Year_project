# Hugging Face Spaces image. Staged and uploaded by scripts/deploy_hf_space.py,
# which copies this file to the root of the Space next to the app, model and
# metrics. The repo-root Dockerfile is the local/clone-to-demo one; this
# differs only where Spaces requires it:
#   - Spaces run containers as uid 1000, so the app directory must be owned
#     by that user or the upload folder and audit.db cannot be written.
#   - The checkpoint is baked in (the Space has no volume to mount).
#   - PNEUMOSCAN_PUBLIC=1: the audit log is world-readable here, so it shows
#     only a prefix of each case uid (a full uid opens that case's upload).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user
ENV HOME=/home/user \
    PNEUMOSCAN_PUBLIC=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

EXPOSE 5000
CMD ["python", "app/app.py"]
