"""Deploy the case reader to Google Cloud Run.

    gcloud auth login
    python scripts/deploy_cloud_run.py --project <gcp-project-id>

Stages the same files as the Hugging Face deploy (app/, src/, configs/, the
checkpoint, metrics.json) with deploy/cloudrun/Dockerfile, then runs
`gcloud run deploy --source`, which builds the image with Cloud Build.

Service settings, and why:
  --memory 2Gi       measured ~850 MB peak with Grad-CAM alone; headroom for
                     Score-CAM/IG/SHAP batches.
  --max-instances 1  uploads and their heatmaps live on the instance that
                     received them; a second instance would answer
                     /explain/<uid> with "unknown case uid". One instance also
                     caps the bill.
  --min-instances 0  scale to zero when idle: free-tier friendly, at the cost
                     of a ~30-60 s cold start for the first visitor.
  --cpu-boost        extra CPU during startup, which is TensorFlow import plus
                     model load.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from deploy_hf_space import REPO_ROOT, stage  # noqa: E402

DOCKERFILE = os.path.join(REPO_ROOT, 'deploy', 'cloudrun', 'Dockerfile')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--project', required=True)
    ap.add_argument('--region', default='asia-south1')
    ap.add_argument('--service', default='pneumoscan-ai')
    args = ap.parse_args()

    gcloud = shutil.which('gcloud') or shutil.which('gcloud.cmd')
    if not gcloud:
        sys.exit('gcloud not found -- install the Google Cloud CLI first')

    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        os.remove(os.path.join(tmp, 'README.md'))          # the HF Space card
        shutil.copy2(DOCKERFILE, os.path.join(tmp, 'Dockerfile'))
        # Without this, gcloud falls back to .gitignore rules and would drop
        # the checkpoint from the upload.
        with open(os.path.join(tmp, '.gcloudignore'), 'w') as f:
            f.write('__pycache__/\n*.pyc\n')
        cmd = [gcloud, 'run', 'deploy', args.service, '--source', tmp,
               '--project', args.project, '--region', args.region,
               '--memory', '2Gi', '--cpu', '2', '--cpu-boost', '--timeout', '600',
               '--min-instances', '0', '--max-instances', '1',
               '--allow-unauthenticated', '--quiet']
        print(' '.join(cmd))
        sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
