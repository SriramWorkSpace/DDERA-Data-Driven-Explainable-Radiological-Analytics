"""Concept-bottleneck composition (ARCHITECTURE stage 6/8) and the model factory.

``ConceptBottleneckModel`` wires ``encoder -> concept_head -> reasoner``. The reasoner
receives **only** the concept vector (Invariant 4). The M1 / M2 / M3 variants differ *not*
in this graph but in what the reasoner is trained against and whether the target gradient
flows back into the concept head:

* **M1 independent** -- reasoner trained on the ground-truth concept vector.
* **M2 sequential**  -- reasoner trained on the *predicted* concepts, detached, so the
  target loss cannot reshape the concept head.
* **M3 joint** (Phase 4) -- predicted concepts, not detached; ``L = L_target + lambda L_concept``.

``build_model`` returns the right object for a :class:`~ddera.config.ModelConfig`.
"""

from __future__ import annotations

import torch
from torch import nn

from ddera.config import ModelConfig
from ddera.models.blackbox import BlackBoxModel
from ddera.models.concept_head import ConceptHead
from ddera.models.reasoner import LinearReasoner

CONCEPTS_FROM = ("predicted", "ground_truth")


class ConceptBottleneckModel(nn.Module):
    """``encoder -> concept_head -> reasoner``, forward over a dataloader batch dict."""

    def __init__(
        self,
        concept_head: ConceptHead,
        reasoner: LinearReasoner,
        *,
        encoder: nn.Module | None = None,
        concepts_from: str = "predicted",
        detach_reasoner_input: bool = True,
    ) -> None:
        super().__init__()
        if concepts_from not in CONCEPTS_FROM:
            raise ValueError(f"concepts_from must be one of {CONCEPTS_FROM}, got {concepts_from!r}")
        if concept_head.n_concepts != reasoner.n_concepts:
            raise ValueError(
                f"concept head produces {concept_head.n_concepts} concepts but the reasoner "
                f"expects {reasoner.n_concepts}."
            )
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.concept_head = concept_head
        self.reasoner = reasoner
        self.concepts_from = concepts_from
        self.detach_reasoner_input = detach_reasoner_input

    @property
    def n_concepts(self) -> int:
        return self.reasoner.n_concepts

    def features(self, batch: dict) -> torch.Tensor:
        """Cached-feature path uses ``batch['features']``; image path runs the encoder."""
        if "features" in batch:
            return batch["features"]
        return self.encoder(batch["image"])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        features = self.features(batch)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)

        if self.concepts_from == "ground_truth":
            reasoner_input = batch["concepts"]  # M1: labels carry no gradient anyway
        else:
            reasoner_input = concept_probs.detach() if self.detach_reasoner_input else concept_probs

        return {
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            # regime-dependent, used for the training loss:
            "target_logit": self.reasoner(reasoner_input),
            # always from the predicted concepts -- the actual inference pathway, used for
            # evaluation and the XAI protocol (matters for M1, where training uses GT concepts):
            "inference_target_logit": self.reasoner(concept_probs),
        }


def build_model(cfg: ModelConfig, *, encoder: nn.Module | None = None) -> nn.Module:
    """Construct the model for ``cfg.variant``.

    ``encoder=None`` gives an ``nn.Identity`` -- the cached-feature path (M1 / M2 / M4). Pass
    a real encoder for B0 / M3, which fine-tune it.
    """
    if cfg.variant == "b0":
        return BlackBoxModel(encoder=encoder, in_features=cfg.encoder.feature_dim)

    head = ConceptHead(cfg.encoder.feature_dim, cfg.n_concepts, dropout=cfg.concept_head_dropout)
    reasoner = LinearReasoner(cfg.n_concepts, bias=cfg.reasoner_bias)

    if cfg.variant == "m1_independent":
        concepts_from, detach = "ground_truth", False
    elif cfg.variant == "m2_sequential":
        concepts_from, detach = "predicted", True
    elif cfg.variant == "m3_joint":
        concepts_from, detach = "predicted", False
    else:  # m4_hybrid
        raise NotImplementedError(
            "m4_hybrid (residual channel) is Phase 4; see ARCHITECTURE section 5 and the "
            "completeness curve. build_model supports b0 / m1_independent / m2_sequential / "
            "m3_joint."
        )

    return ConceptBottleneckModel(
        head,
        reasoner,
        encoder=encoder,
        concepts_from=concepts_from,
        detach_reasoner_input=detach,
    )


class EncoderWrapped(nn.Module):
    """Evaluate a frozen-encoder model (built with ``encoder=Identity`` for the cache path)
    on **image** batches, by running ``encoder`` first. Used for the stability family, which
    needs concept vectors re-inferred from perturbed images.
    """

    def __init__(self, encoder: nn.Module, model: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.model = model

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if "features" not in batch and "image" in batch:
            batch = {**batch, "features": self.encoder(batch["image"])}
        return self.model(batch)

    @property
    def reasoner(self):  # passthrough so evaluate_model can read the weights
        return self.model.reasoner
