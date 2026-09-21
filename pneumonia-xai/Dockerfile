# Two-command clone-to-demo (G-9):
#   docker build -t pneumonia-xai .
#   docker run -p 5000:5000 -v "$(pwd)/models:/app/models" pneumonia-xai
#
# The dataset is NOT baked into the image (2.4 GB, and not this repo's to
# redistribute -- see docs/dataset_datasheet.md). Mount a trained
# models/pneumonia_model.h5 (and models/metrics.json if you've run the eval
# suite) as shown above; the app runs fine without metrics.json, it just
# shows the dashboard's "not yet evaluated" state.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV FLASK_ENV=production
EXPOSE 5000

# /healthz loads the checkpoint and reports whether it is usable, so an image
# started without a mounted model is reported unhealthy rather than serving a
# page that 500s on the first upload.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3     CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/healthz').status==200 else 1)"

CMD ["python", "app/app.py"]
