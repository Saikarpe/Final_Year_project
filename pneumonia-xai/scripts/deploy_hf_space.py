"""Deploy the case reader to a Hugging Face Space (Docker SDK).

    hf auth login                      # once; needs a token with write access
    python scripts/deploy_hf_space.py --space <username>/pneumoscan-ai

Stages exactly what the running app reads -- app/, src/, configs/, the
checkpoint and metrics.json -- plus deploy/hf-space/{Dockerfile,README.md}
into a temporary folder, then uploads it. The checkpoint is gitignored, so
it is taken from models/ on this machine; the Space's git repo stores it via
LFS. The dataset is never uploaded (see docs/dataset_datasheet.md), so the
Space's "Use sample image" button stays hidden.
"""
import argparse
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SPACE_DIR = os.path.join(REPO_ROOT, 'deploy', 'hf-space')

TREES = ['app', 'src', 'configs']
FILES = ['requirements.txt', 'pyproject.toml', 'models/pneumonia_model.h5', 'models/metrics.json']
# Runtime state and caches from local runs must not ship: uploads are other
# people's X-rays, and audit.db is this machine's log.
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', 'audit.db')


def stage(dest: str) -> None:
    for tree in TREES:
        shutil.copytree(os.path.join(REPO_ROOT, tree), os.path.join(dest, tree), ignore=IGNORE)
    uploads = os.path.join(dest, 'app', 'static', 'uploads')
    shutil.rmtree(uploads, ignore_errors=True)
    os.makedirs(uploads)
    open(os.path.join(uploads, '.gitkeep'), 'w').close()

    for rel in FILES:
        src = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(src):
            sys.exit(f'missing {rel} -- train/evaluate first (see README run order)')
        os.makedirs(os.path.dirname(os.path.join(dest, rel)) or dest, exist_ok=True)
        shutil.copy2(src, os.path.join(dest, rel))
    for name in ('Dockerfile', 'README.md'):
        shutil.copy2(os.path.join(SPACE_DIR, name), os.path.join(dest, name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--space', required=True, help='<username>/<space-name>')
    ap.add_argument('--private', action='store_true', help='create the Space as private')
    ap.add_argument('--dry-run', action='store_true', help='stage only, print the file list')
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        if args.dry_run:
            for root, _, files in os.walk(tmp):
                for f in files:
                    p = os.path.join(root, f)
                    print(f'{os.path.getsize(p):>10,}  {os.path.relpath(p, tmp)}')
            return

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.space, repo_type='space', space_sdk='docker',
                        private=args.private, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type='space',
                          commit_message='Deploy PneumoScan AI',
                          delete_patterns=['*'])  # mirror: drop files no longer staged
    print(f'Deployed. Build log and app: https://huggingface.co/spaces/{args.space}')


if __name__ == '__main__':
    main()
