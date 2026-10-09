# Render image, built from the GitHub repo by ../../../render.yaml with
# pneumonia-xai/ as the build context.
#
# Tuned for Render's 512 MB free/starter instances. The app measured ~570 MB
# idle-after-inference and ~850 MB peak on Windows with all seven methods, so
# this image trims everything it can and may still be OOM-killed:
#   - tensorflow-cpu instead of tensorflow (same 2.13 API, no CUDA payload);
#   - no shap (the registry already degrades to the other methods without it)
#     and no pytest;
#   - render.yaml limits methods to the three cheap ones and sets the
#     allocator/thread env vars below.
# The checkpoint is gitignored, so start.sh downloads it at boot from a
# private Hugging Face model repo using the HF_TOKEN secret.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN sed -e 's/^tensorflow==/tensorflow-cpu==/' -e '/^shap==/d' -e '/^pytest==/d' requirements.txt > req-render.txt \
    && pip install --no-cache-dir -r req-render.txt huggingface_hub==0.36.2

COPY . .
RUN chmod +x deploy/render/start.sh

ENV PNEUMOSCAN_PUBLIC=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1

CMD ["deploy/render/start.sh"]
