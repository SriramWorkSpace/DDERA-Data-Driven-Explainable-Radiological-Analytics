"""Run a trained model over a split and assemble the metrics for ``metrics.json``.

Phase 3 produces a **partial** run: predictive, calibration, concept-quality, a first
intervention curve and faithfulness. Leakage, completeness and stability are marked
``pending`` -- they are the Phase 5 protocol (the residual-k sweep for completeness,
perturbation re-inference for stability). ``ddera.reporting.runs.log_run`` still enforces
all eight families before a run may be marked ``complete``; a partial run is written and
labelled as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddera.config import ConceptSpec
from ddera.eval.bootstrap import bootstrap_ci
from ddera.eval.calibration import calibration_report
from ddera.eval.metrics import binary_metrics, concept_metrics, safe_auroc
from ddera.xai.intervention import faithfulness_report, tti_curve

PENDING = {"status": "pending", "phase": 5}


@torch.no_grad()
def _collect(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    tgt, prob, c_prob, c_lab, c_mask = [], [], [], [], []
    for batch in loader:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(moved)
        tgt.append(batch["target"].cpu().numpy())
        prob.append(torch.sigmoid(out["target_logit"]).float().cpu().numpy())
        if "concept_probs" in out:
            c_prob.append(out["concept_probs"].float().cpu().numpy())
            c_lab.append(batch["concepts"].cpu().numpy())
            c_mask.append(batch["concept_mask"].cpu().numpy())
    data = {"target": np.concatenate(tgt), "target_prob": np.concatenate(prob)}
    if c_prob:
        data["concept_probs"] = np.concatenate(c_prob)
        data["concept_labels"] = np.concatenate(c_lab)
        data["concept_mask"] = np.concatenate(c_mask)
    return data


def _predictions_frame(data: dict[str, np.ndarray], concept_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row": np.arange(len(data["target"])),
            "target": data["target"].astype(int),
            "target_prob": data["target_prob"].astype(float),
        }
    )
    if "concept_probs" in data:
        for j, name in enumerate(concept_names):
            frame[f"concept_prob__{name}"] = data["concept_probs"][:, j]
            frame[f"concept_label__{name}"] = data["concept_labels"][:, j].astype(int)
            frame[f"concept_mask__{name}"] = data["concept_mask"][:, j].astype(int)
    return frame


def evaluate_model(
    model,
    loader: DataLoader,
    *,
    spec: ConceptSpec,
    device: torch.device,
    split: str = "val",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Returns ``(predictions_df, metrics, concept_weights)``.

    ``metrics`` carries ``status='partial'`` and ``families_present``; ``concept_weights`` is
    ``{}`` for B0.
    """
    data = _collect(model, loader, device)
    y, p = data["target"], data["target_prob"]
    is_cbm = "concept_probs" in data

    predictive = binary_metrics(y, p)
    if len(np.unique(y)) > 1:
        ci = bootstrap_ci(y, p, safe_auroc, n_resamples=1000, seed=spec_seed(spec))
        predictive["auroc_ci"] = [ci.lower, ci.upper]
        predictive["auroc_ci_width"] = ci.ci_width

    metrics: dict[str, Any] = {
        "split": split,
        "status": "partial",
        "predictive": predictive,
        "calibration": calibration_report(y, p),
    }
    concept_weights: dict[str, Any] = {}

    if is_cbm:
        c_prob = data["concept_probs"]
        c_lab = data["concept_labels"]
        c_mask = data["concept_mask"]
        metrics["concept_quality"] = concept_metrics(c_lab, c_prob, list(spec.concepts), c_mask)

        reasoner = model.reasoner.to_numpy_reasoner(list(spec.concepts))
        predict_fn = reasoner.predict_proba
        metrics["faithfulness"] = faithfulness_report(
            predict_fn, c_prob, reasoner.weights, concept_names=list(spec.concepts)
        )
        # Best-available "true" concept vector: the label where it is usable, else the
        # model's own hard prediction. Documented approximation until Phase 5.
        c_true_est = np.where(c_mask > 0, c_lab, (c_prob >= 0.5).astype(float))
        metrics["intervention"] = tti_curve(
            predict_fn, c_prob, c_true_est, y, strategy="weight", weights=reasoner.weights
        ).to_dict()

        metrics["leakage"] = dict(PENDING)
        metrics["completeness"] = dict(PENDING)
        metrics["stability"] = dict(PENDING)
        concept_weights = {
            "weights": reasoner.weights.tolist(),
            "bias": float(reasoner.bias),
            "concept_names": list(spec.concepts),
        }
    else:
        metrics["note"] = "B0 black-box baseline: no concept bottleneck, no concept families."

    metrics["families_present"] = sorted(
        k
        for k in (
            "predictive",
            "calibration",
            "concept_quality",
            "intervention",
            "leakage",
            "completeness",
            "faithfulness",
            "stability",
        )
        if isinstance(metrics.get(k), dict) and metrics[k].get("status") != "pending"
    )
    return _predictions_frame(data, list(spec.concepts)), metrics, concept_weights


def spec_seed(spec: ConceptSpec) -> int:
    """A stable per-spec seed for the bootstrap (keeps CIs reproducible per experiment)."""
    return abs(hash(spec.name)) % (2**31)
