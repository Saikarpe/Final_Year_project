"""Thin wrapper kept for the README's existing run order.
All real logic lives in xai_cxr / scripts/train.py -- zero duplication (G-3)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train.py'), run_name='__main__')
