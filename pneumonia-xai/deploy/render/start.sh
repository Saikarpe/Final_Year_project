#!/bin/sh
# Fetch the checkpoint (gitignored, so not in the build context) and start the app.
set -e

if [ ! -f models/pneumonia_model.h5 ]; then
  if [ -z "$MODEL_REPO" ]; then
    echo "MODEL_REPO is not set; starting without a model (the UI will say so)." >&2
  else
    echo "Downloading checkpoint from $MODEL_REPO ..."
    python -c "
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ['MODEL_REPO'], 'pneumonia_model.h5', token=os.environ.get('HF_TOKEN') or None)
shutil.copy(p, 'models/pneumonia_model.h5')
print('checkpoint ready:', os.path.getsize('models/pneumonia_model.h5'), 'bytes')
"
  fi
fi

exec python app/app.py
