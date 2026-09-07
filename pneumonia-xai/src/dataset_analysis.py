"""Thin wrapper kept for backward compatibility.
Real logic lives in scripts/dataset_analysis.py -- zero duplication (G-3)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'dataset_analysis.py'), run_name='__main__')
