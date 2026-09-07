"""Lightweight experiment tracking (G-8): every run logged with its config,
into a results table anyone on the team can read without extra infra.

Not MLflow/W&B — the spec's actual acceptance criterion is "every run logged
with its config" + "a results table anyone on the team can read", and a
runs/<timestamp>/ folder plus one CSV satisfies that with zero new services.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
from datetime import datetime, timezone

from .config import RUNS_DIR


def start_run(name: str, config: dict) -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = os.path.join(RUNS_DIR, f'{ts}_{name}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)  # valid YAML is valid JSON; keeps this dependency-free
    return run_dir


def log_metrics(run_dir: str, metrics: dict) -> None:
    with open(os.path.join(run_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    _append_results_row(run_dir, metrics)


def _append_results_row(run_dir: str, metrics: dict) -> None:
    path = os.path.join(RUNS_DIR, 'results_table.csv')
    row = {'run': os.path.basename(run_dir), 'logged_at': datetime.now(timezone.utc).isoformat()}
    row.update({k: v for k, v in _flatten(metrics).items()})
    write_header = not os.path.exists(path)
    # Union existing header with any new keys so the table never truncates columns.
    existing_fields: list[str] = []
    existing_rows: list[dict] = []
    if not write_header:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing_rows = list(reader)
    fields = list(dict.fromkeys(existing_fields + list(row.keys())))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(row)


def _flatten(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f'{key}.'))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out
