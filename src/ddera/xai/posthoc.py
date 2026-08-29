"""Post-hoc explanation methods -- BLACK-BOX BASELINES ONLY.

.. warning::

   **Invariant 5.** Grad-CAM, SHAP and LIME exist in this project purely as comparison
   baselines against the black-box model ``B0``. They are never DDERA's explanation
   mechanism, and they must never be applied to the CBM path or shown in the dashboard's
   explanation panel.

   This is enforced in code, not merely documented: every entry point calls
   :func:`assert_baseline_only`, which refuses any model carrying the ``is_ante_hoc``
   marker set by :mod:`ddera.models.cbm`.

**Why they are baselines rather than the method.** The comparison DDERA draws is not
"our saliency map looks nicer". It is structural:

===================================  ==================  ==========================
Property                             Post-hoc saliency   DDERA concept bottleneck
===================================  ==================  ==========================
Influenced the prediction            No                  Yes -- it *is* the pathway
Can be intervened upon               No                  Yes
Exact, not approximate               No                  Yes (``logit = b + sum w_j c_j``)
Quantitatively falsifiable           Barely              Yes (Phase 5 protocol)
Expressed in clinical vocabulary     No (pixels)         Yes (named concepts)
===================================  ==================  ==========================

A saliency map cannot answer "if this consolidation were absent, would the diagnosis
change?". That question is the point of the project, and only the concept pathway can
answer it. Phase 7 makes that argument with evidence rather than assertion.

SHAP and LIME are imported lazily so the package works without them installed; they are not
needed before Phase 7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

BASELINE_ONLY_MESSAGE = (
    "Invariant 5 violation: post-hoc explanation methods are comparison baselines for the "
    "black-box model only. They must never be applied to the concept-bottleneck path, whose "
    "explanation is intrinsic and exact. If you need to explain the CBM, use "
    "ddera.xai.intervention.LinearReasoner.contributions()."
)


def assert_baseline_only(model: Any) -> None:
    """Refuse to run a post-hoc method on an ante-hoc model.

    Models built by :mod:`ddera.models.cbm` carry ``is_ante_hoc = True``. This turns
    Invariant 5 from a convention into a runtime error, so the mistake cannot survive to a
    results table.
    """
    if getattr(model, "is_ante_hoc", False):
        raise RuntimeError(BASELINE_ONLY_MESSAGE)


class GradCAM:
    """Grad-CAM for a black-box CNN baseline.

    Implemented directly in PyTorch rather than pulled from a dependency: it is a dozen
    lines, and owning it keeps the baseline reproducible and pinned to nothing.

    Produces a class-activation map by weighting a convolutional layer's feature maps by the
    gradient of the target logit with respect to them, then applying ReLU.

    Example:
        >>> cam = GradCAM(blackbox_model, blackbox_model.features.denseblock4)  # doctest: +SKIP
        >>> heatmap = cam(images)                                               # doctest: +SKIP
        >>> cam.close()                                                         # doctest: +SKIP
    """

    def __init__(self, model: Any, target_layer: Any) -> None:
        assert_baseline_only(model)
        self.model = model
        self.target_layer = target_layer
        self._activations: Any = None
        self._gradients: Any = None
        self._handles: list[Any] = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module: Any, _grad_in: Any, grad_out: Any) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, inputs: Any, target_index: int | None = None) -> npt.NDArray[np.float64]:
        """Return a normalised ``(batch, height, width)`` heatmap."""
        import torch

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(inputs)
        if outputs.ndim == 1:
            outputs = outputs[:, None]
        scores = outputs[:, target_index] if target_index is not None else outputs[:, 0]
        scores.sum().backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations captured. Check that `target_layer` is actually used in the "
                "model's forward pass."
            )

        # One scalar weight per channel: the mean gradient over spatial positions.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1))

        # Normalise per sample so maps are comparable across images.
        flat = cam.flatten(1)
        lo = flat.min(dim=1).values[:, None, None]
        hi = flat.max(dim=1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        """Remove hooks. Forgetting this leaks memory across a long evaluation loop."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def shap_values(
    model: Any,
    background: npt.ArrayLike,
    samples: npt.ArrayLike,
    *,
    n_samples: int = 100,
) -> npt.NDArray[np.float64]:
    """SHAP values for the black-box baseline. Requires ``pip install shap``.

    Note the cost asymmetry worth reporting in Phase 7: this needs many model evaluations
    per explanation and yields an *approximation*, whereas the CBM's contributions are one
    multiplication and exact.
    """
    assert_baseline_only(model)
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional Phase 7 dependency
        raise ImportError(
            "SHAP is not installed. It is only needed for Phase 7 baselines: pip install shap"
        ) from exc

    explainer = shap.KernelExplainer(model, np.asarray(background))
    return np.asarray(explainer.shap_values(np.asarray(samples), nsamples=n_samples))


def lime_explanation(
    model: Any,
    image: npt.ArrayLike,
    *,
    n_samples: int = 1000,
    top_labels: int = 1,
) -> Any:
    """LIME explanation for the black-box baseline. Requires ``pip install lime``."""
    assert_baseline_only(model)
    try:
        from lime import lime_image
    except ImportError as exc:  # pragma: no cover - optional Phase 7 dependency
        raise ImportError(
            "LIME is not installed. It is only needed for Phase 7 baselines: pip install lime"
        ) from exc

    explainer = lime_image.LimeImageExplainer()
    return explainer.explain_instance(
        np.asarray(image), model, top_labels=top_labels, num_samples=n_samples
    )


def comparison_table() -> dict[str, dict[str, Any]]:
    """The structural comparison Phase 7 reports, as data rather than prose."""
    return {
        "post_hoc_saliency": {
            "influenced_prediction": False,
            "interveneable": False,
            "exact": False,
            "quantitatively_falsifiable": False,
            "clinical_vocabulary": False,
            "cost_per_explanation": "many model evaluations",
        },
        "ddera_concept_bottleneck": {
            "influenced_prediction": True,
            "interveneable": True,
            "exact": True,
            "quantitatively_falsifiable": True,
            "clinical_vocabulary": True,
            "cost_per_explanation": "one elementwise multiplication",
        },
    }
