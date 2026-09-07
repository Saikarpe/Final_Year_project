"""Thin wrapper kept for the README's existing run order.
Grad-CAM now lives in xai_cxr.explain.gradcam, registered alongside five other
methods in xai_cxr.explain.registry -- see scripts/explain.py, which renders
all of them (zero duplication, G-3)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'explain.py'), run_name='__main__')
