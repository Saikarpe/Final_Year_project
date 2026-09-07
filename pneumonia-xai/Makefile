.PHONY: setup train eval explain audit serve test splits

# Windows note: this Makefile assumes a Unix-like shell (Git Bash/WSL/macOS/
# Linux). Without `make` available, run the same commands directly --
# each target below is exactly one `python ...` invocation.

setup:
	pip install -e ".[dev]"

splits:
	python scripts/build_splits.py

train: splits
	python scripts/train.py

eval:
	python scripts/evaluate.py

explain:
	python scripts/explain.py

audit:
	python scripts/audit.py

serve:
	cd app && python app.py

test:
	pytest tests/ -v
