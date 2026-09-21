"""Attention rollout (X-7): the head's own pooling weights as a saliency map.

Every other method in the registry is *post-hoc* -- it reconstructs, after the
fact, which inputs a decision leaned on. This one is not. When the model is
built with `head: attention` (see xai_cxr.models.heads.AttentionPool2D), the
weights it shows are literally the coefficients the classifier used to combine
the feature grid into the vector it scored. There is no approximation step to
be unfaithful.

That makes it the cheapest honest baseline in the suite -- one forward pass,
no backward pass, no perturbation loop -- and a useful control: where Grad-CAM
and the attention map disagree, one of them is wrong about the same model.

It is not free of caveats. Attention weights say which grid cells were
*pooled*, not how the logit would move if a cell changed, so they carry no
sign and can highlight a region the head then weights negatively. The
faithfulness suite (deletion/insertion/ROAD) scores it against the others on
exactly that basis rather than taking the architecture's word for it.
"""
from __future__ import annotations

import numpy as np

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def is_available(xai_model: XAIModel) -> bool:
    return xai_model.has_attention


def explain(xai_model: XAIModel, image: np.ndarray,
            img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    if not xai_model.has_attention:
        raise ValueError(
            "The 'attention' method needs a model trained with head: attention "
            'in configs/model.yaml. Use registry.available_methods(xai_model) to '
            'list the methods a given model actually supports.')
    images = as_batch(image)
    alpha = xai_model.attention_map(images)[0]     # (h, w), sums to 1
    return resize_to(normalize(alpha), img_size)
