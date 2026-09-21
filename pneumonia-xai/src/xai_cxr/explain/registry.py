"""Method registry (X-9): name -> implementation, so the evaluation suite
(§6) and the app's method selector both just iterate this dict instead of
hardcoding a method list. Adding a seventh method later is a one-line change
here, nowhere else.
"""
from __future__ import annotations

import numpy as np

from ..config import ExplainConfig
from ..models.baseline import XAIModel
from . import attention, gradcam, gradcampp, scorecam, integrated_gradients, occlusion

# SHAP is the only method with a heavy third-party dependency, and it is the
# weakest performer in the benchmark matrix. Importing it at module scope made
# `import xai_cxr.explain` fail outright when `shap` was missing -- which took
# down evaluate.py, explain.py and audit.py entirely rather than costing one
# method out of seven. Degrade to six methods and say so, once.
try:
    from . import shap_method
    _SHAP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    shap_method = None
    _SHAP_IMPORT_ERROR = exc
    import warnings
    warnings.warn(
        f'SHAP unavailable ({exc}); continuing with the other explanation methods. '
        'Install `shap` to include it.', RuntimeWarning, stacklevel=2)

METHODS = {
    'attention': attention.explain,
    'gradcam': gradcam.explain,
    'gradcam++': gradcampp.explain,
    'scorecam': scorecam.explain,
    'integrated_gradients': integrated_gradients.explain,
    'occlusion': occlusion.explain,
}
if shap_method is not None:
    METHODS['shap'] = shap_method.explain

# Kept honest and visible in the UI/eval-suite output — matches the spec's
# per-method notes in §4 rather than presenting all six as interchangeable.
METHOD_INFO = {
    'attention': {
        'label': 'Attention (head)',
        'kind': 'intrinsic',
        'notes': "Fastest -- one forward pass. The head's actual pooling weights, not a "
                 'post-hoc reconstruction. Unsigned: shows what was pooled, not which way it pushed. '
                 'Requires head: attention.',
    },
    'gradcam': {
        'label': 'Grad-CAM',
        'kind': 'gradient',
        'notes': 'Fast. The incumbent method; backprops from the pre-activation logit.',
    },
    'gradcam++': {
        'label': 'Grad-CAM++',
        'kind': 'gradient',
        'notes': 'Fast. Better than Grad-CAM when a pathology has multiple/overlapping instances.',
    },
    'scorecam': {
        'label': 'Score-CAM',
        'kind': 'gradient-free',
        'notes': 'Slow (one forward pass per conv channel). Sidesteps gradient saturation entirely.',
    },
    'integrated_gradients': {
        'label': 'Integrated Gradients',
        'kind': 'axiomatic',
        'notes': 'Moderate cost. Attributions sum to the logit delta from baseline (completeness).',
    },
    'occlusion': {
        'label': 'Occlusion',
        'kind': 'perturbation',
        'notes': 'Slowest. Model-agnostic; used as a ground-truth-ish reference.',
    },
    'shap': {
        'label': 'SHAP',
        'kind': 'gradient (Shapley-approximate)',
        'notes': 'Slow. Kept for comparison; weaker fidelity on correlated X-ray pixels.',
    },
}


def available_methods(xai_model: XAIModel | None = None) -> list[str]:
    """The methods a *given model* supports, in registry order.

    'attention' only exists when the model was built with an attention-pooling
    head, so anything that iterates the registry (the app's method selector,
    the runtime timings in the eval suite) must filter through here rather
    than assuming every registered method applies to every checkpoint.
    """
    names = [n for n in METHODS if n in METHOD_INFO]
    if xai_model is not None and not attention.is_available(xai_model):
        names.remove('attention')
    return names


def config_kwargs(cfg: ExplainConfig, method: str) -> dict:
    """The speed/quality knobs configs/explain.yaml controls for a given
    method. Every caller (evaluate.py, explain.py, the app) should route
    through this rather than hardcoding patch sizes, channel counts, etc. --
    that hardcoding is exactly what made the original evaluate.py run take
    17+ hours on CPU (see docs/decisions_log.md)."""
    if method == 'occlusion':
        return {'patch': cfg.occlusion_patch, 'stride': cfg.occlusion_stride,
                'batch_size': cfg.inference_batch}
    if method == 'scorecam':
        return {'batch_size': cfg.inference_batch, 'max_channels': cfg.scorecam_max_channels}
    if method == 'integrated_gradients':
        return {'steps': cfg.ig_steps, 'chunk_size': cfg.inference_batch}
    if method == 'shap':
        return {'background_size': cfg.shap_background_size,
                'batch_size': cfg.inference_batch}
    return {}


def explain(method: str, xai_model: XAIModel, image: np.ndarray,
            explain_cfg: ExplainConfig | None = None, **kwargs) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f'Unknown explanation method {method!r}. Available: {sorted(METHODS)}')
    call_kwargs = config_kwargs(explain_cfg or ExplainConfig.load(), method)
    call_kwargs.update(kwargs)  # explicit kwargs always win over config defaults
    return METHODS[method](xai_model, image, **call_kwargs)
