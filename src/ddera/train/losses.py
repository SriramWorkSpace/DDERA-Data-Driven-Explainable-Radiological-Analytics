"""Training losses.

* :func:`masked_bce_with_logits` -- concept loss under ADR-004 ``u_mask``: a per-element
  0/1 mask, masked entries contribute no loss and therefore no gradient.
* :func:`target_bce` -- the target loss.
* :func:`cbm_loss` -- combines them per training regime (M1 / M2 / M3). The regime's
  gradient behaviour (detach vs joint) lives in
  :class:`ddera.models.cbm.ConceptBottleneckModel`; this function only weights the terms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

REGIMES = ("independent", "sequential", "joint")


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE-with-logits, per element, times ``mask``.

    ``reduction='mean'`` divides by the number of **unmasked** elements (clamped to >= 1), so
    a batch with few valid concept labels is not silently down-weighted to zero.
    """
    per_element = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    per_element = per_element * mask
    if reduction == "sum":
        return per_element.sum()
    if reduction == "none":
        return per_element
    return per_element.sum() / mask.sum().clamp_min(1.0)


def target_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def cbm_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    regime: str,
    concept_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Total loss for a CBM training step.

    Returns ``{"loss", "concept_loss", "target_loss"}``; the last two are detached scalars
    for logging.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")

    concept_loss = masked_bce_with_logits(
        outputs["concept_logits"], batch["concepts"], batch["concept_mask"]
    )
    target_loss = target_bce(outputs["target_logit"], batch["target"])

    if regime == "joint":
        total = target_loss + concept_weight * concept_loss
    else:  # independent, sequential -- both heads carry their own term at weight 1
        total = concept_loss + target_loss

    return {
        "loss": total,
        "concept_loss": concept_loss.detach(),
        "target_loss": target_loss.detach(),
    }
