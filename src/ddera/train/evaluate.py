"""Run a trained model over a split, write ``predictions.parquet``, assemble ``metrics.json``.

The metric assembly is :func:`ddera.xai.protocol.run_protocol`, which works off the
predictions frame alone (ARCHITECTURE 6/8). This module's job is to *produce* that frame:
run the model, and -- when image datasets are supplied -- re-infer the concept vector under
a few fixed input perturbations so the stability family can be computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ddera.config import ConceptSpec
from ddera.xai.protocol import FAMILIES, run_protocol


@torch.no_grad()
def _concept_probs(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out.append(torch.sigmoid(model(moved)["concept_logits"]).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _collect(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    tgt, prob, c_prob, c_lab, c_mask = [], [], [], [], []
    for batch in loader:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        result = model(moved)
        tgt.append(batch["target"].cpu().numpy())
        # The inference pathway: predicted concepts -> reasoner (not the training logit).
        logit = result.get("inference_target_logit", result["target_logit"])
        prob.append(torch.sigmoid(logit).float().cpu().numpy())
        if "concept_probs" in result:
            c_prob.append(result["concept_probs"].float().cpu().numpy())
            c_lab.append(batch["concepts"].cpu().numpy())
            c_mask.append(batch["concept_mask"].cpu().numpy())
    data = {"target": np.concatenate(tgt), "target_prob": np.concatenate(prob)}
    if c_prob:
        data["concept_probs"] = np.concatenate(c_prob)
        data["concept_labels"] = np.concatenate(c_lab)
        data["concept_mask"] = np.concatenate(c_mask)
    return data


def _predictions_frame(
    data: dict[str, np.ndarray],
    concept_names: list[str],
    perturbed: dict[str, np.ndarray] | None,
) -> pd.DataFrame:
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
        for pert_name, matrix in (perturbed or {}).items():
            for j, name in enumerate(concept_names):
                frame[f"concept_prob__{name}__{pert_name}"] = matrix[:, j]
    return frame


def evaluate_model(
    model,
    loader: DataLoader,
    *,
    spec: ConceptSpec,
    device: torch.device,
    split: str = "val",
    perturbation_loaders: dict[str, DataLoader] | None = None,
    features: np.ndarray | None = None,
    baseline_predictions: pd.DataFrame | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Returns ``(predictions_df, metrics, concept_weights)``.

    ``perturbation_loaders`` (name -> loader over the same split with a perturbing transform)
    enables the stability family. ``baseline_predictions`` (a B0 run's frame) enables the
    completeness interpretability-cost. ``metrics['status']`` is ``partial`` unless every
    family is present and non-partial.
    """
    data = _collect(model, loader, device)
    is_cbm = "concept_probs" in data
    names = list(spec.concepts)

    if not is_cbm:  # B0 black box -- no concept families
        predictions = pd.DataFrame(
            {
                "row": np.arange(len(data["target"])),
                "target": data["target"].astype(int),
                "target_prob": data["target_prob"].astype(float),
            }
        )
        from ddera.eval.bootstrap import bootstrap_ci
        from ddera.eval.calibration import calibration_report
        from ddera.eval.metrics import binary_metrics, safe_auroc

        predictive = binary_metrics(data["target"], data["target_prob"])
        if len(np.unique(data["target"])) > 1:
            ci = bootstrap_ci(data["target"], data["target_prob"], safe_auroc, seed=seed)
            predictive["auroc_ci"] = [ci.lower, ci.upper]
            predictive["auroc_ci_width"] = ci.ci_width
        metrics = {
            "split": split,
            "status": "partial",
            "predictive": predictive,
            "calibration": calibration_report(data["target"], data["target_prob"]),
            "note": "B0 black-box baseline: no concept bottleneck, no concept families.",
            "families_present": ["predictive", "calibration"],
        }
        return predictions, metrics, {}

    perturbed = {
        name: _concept_probs(model, dl, device) for name, dl in (perturbation_loaders or {}).items()
    }
    predictions = _predictions_frame(data, names, perturbed)

    reasoner = model.reasoner.to_numpy_reasoner(names)
    concept_weights = {
        "weights": reasoner.weights.tolist(),
        "bias": float(reasoner.bias),
        "concept_names": names,
    }

    metrics = run_protocol(
        predictions,
        concept_weights,
        features=features,
        baseline_predictions=baseline_predictions,
        seed=seed,
    )
    metrics["split"] = split
    metrics["families_present"] = _present(metrics)
    metrics["status"] = (
        "complete" if len(metrics["families_present"]) == len(FAMILIES) else "partial"
    )
    return predictions, metrics, concept_weights


def _present(metrics: dict[str, Any]) -> list[str]:
    out = []
    for fam in FAMILIES:
        value = metrics.get(fam)
        if (
            isinstance(value, dict)
            and value.get("status") != "pending"
            and not value.get("partial_curve")
        ):
            out.append(fam)
    return sorted(out)
