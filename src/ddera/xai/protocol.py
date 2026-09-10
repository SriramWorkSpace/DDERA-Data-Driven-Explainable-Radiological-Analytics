"""The eight-family evaluation protocol, assembled from run artifacts (ARCHITECTURE 6/8).

``run_protocol`` takes a ``predictions`` frame and a ``concept_weights`` dict -- exactly what
``ddera.reporting.runs`` writes -- and returns the metrics dict that
``ddera.reporting.runs.log_run`` checks against Invariant 6. Nothing here touches a live
model or a GPU, and nothing here is chest-X-ray specific (Invariants 1/2/9): the inputs are
``(concepts, predictions, labels, reasoner weights, concept names)`` and nothing else.

Families 1-3 are table stakes; 4-8 are the reusable contribution:

1. predictive      binary_metrics + bootstrap AUROC CI
2. calibration     ECE / MCE / Brier / reliability
3. concept_quality per-concept AUROC/AUPRC/F1 under the u_mask mask
4. intervention    TTI curves under all four orderings
5. leakage         permutation necessity + soft-vs-hard + residual probe
6. completeness    bottleneck AUROC (+ interpretability cost vs a baseline; the full
                   residual-k curve is the Phase 4 sweep)
7. faithfulness    measured d logit / d c_j vs the declared w_j
8. stability       concept drift / rank stability / flip rate under input perturbation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ddera.eval.bootstrap import bootstrap_ci
from ddera.eval.calibration import calibration_report
from ddera.eval.metrics import binary_metrics, concept_metrics, safe_auroc
from ddera.xai.completeness import completeness_report
from ddera.xai.intervention import LinearReasoner, faithfulness_report, tti_all_strategies
from ddera.xai.leakage import leakage_report
from ddera.xai.stability import stability_report

FAMILIES = (
    "predictive",
    "calibration",
    "concept_quality",
    "intervention",
    "leakage",
    "completeness",
    "faithfulness",
    "stability",
)


def _concept_columns(predictions: pd.DataFrame, names: list[str], suffix: str = "") -> np.ndarray:
    cols = [f"concept_prob__{n}{suffix}" for n in names]
    return predictions[cols].to_numpy(dtype=float)


def _perturbation_names(predictions: pd.DataFrame) -> list[str]:
    # base column: "concept_prob__<name>"  (one "__");  perturbed: "concept_prob__<name>__<pert>"
    perts: list[str] = []
    for col in predictions.columns:
        if col.startswith("concept_prob__") and col.count("__") == 2:
            perts.append(col.rsplit("__", 1)[1])
    return sorted(set(perts))


def run_protocol(
    predictions: pd.DataFrame,
    concept_weights: dict[str, Any],
    *,
    features: np.ndarray | None = None,
    baseline_predictions: pd.DataFrame | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Assemble every computable metric family. Families that genuinely need more than this
    one run (the completeness *curve*; stability without perturbed columns) are returned as
    partial / pending and will keep the run ``partial``.
    """
    names: list[str] = list(concept_weights["concept_names"])
    weights = np.asarray(concept_weights["weights"], dtype=float)
    bias = float(concept_weights["bias"])
    reasoner = LinearReasoner(weights, bias, names)
    predict_fn = reasoner.predict_proba

    target = predictions["target"].to_numpy(dtype=int)
    target_prob = predictions["target_prob"].to_numpy(dtype=float)
    c_pred = _concept_columns(predictions, names)
    c_label = predictions[[f"concept_label__{n}" for n in names]].to_numpy(dtype=float)
    c_mask = predictions[[f"concept_mask__{n}" for n in names]].to_numpy(dtype=float)
    # Best-available target for interventions: the label where usable, else the hard prediction.
    c_true_est = np.where(c_mask > 0, c_label, (c_pred >= 0.5).astype(float))

    metrics: dict[str, Any] = {}

    # 1. predictive
    predictive = binary_metrics(target, target_prob)
    if len(np.unique(target)) > 1:
        ci = bootstrap_ci(target, target_prob, safe_auroc, n_resamples=1000, seed=seed)
        predictive["auroc_ci"] = [ci.lower, ci.upper]
        predictive["auroc_ci_width"] = ci.ci_width
    metrics["predictive"] = predictive

    # 2. calibration
    metrics["calibration"] = calibration_report(target, target_prob)

    # 3. concept quality
    metrics["concept_quality"] = concept_metrics(c_label, c_pred, names, c_mask)

    # 4. intervention -- all four orderings
    curves = tti_all_strategies(predict_fn, c_pred, c_true_est, target, weights, seed=seed)
    metrics["intervention"] = {name: curve.to_dict() for name, curve in curves.items()}

    # 5. leakage
    metrics["leakage"] = leakage_report(
        predict_fn,
        c_pred,
        target,
        features=features,
        base_probs=target_prob,
        concept_names=names,
        seed=seed,
    )

    # 6. completeness -- this run's bottleneck point; the curve is the Phase 4 sweep
    bottleneck_auroc = safe_auroc(target, target_prob)
    if baseline_predictions is not None and len(baseline_predictions):
        ref_auroc = safe_auroc(
            baseline_predictions["target"].to_numpy(int),
            baseline_predictions["target_prob"].to_numpy(float),
        )
        metrics["completeness"] = completeness_report(
            {0: bottleneck_auroc}, reference_auroc=ref_auroc
        )
    else:
        metrics["completeness"] = {
            "bottleneck_auroc": bottleneck_auroc,
            "reference_auroc": None,
            "partial_curve": True,
            "note": (
                "Interpretability cost needs a B0 baseline run; the full residual-k curve "
                "is the Phase 4 M4 sweep."
            ),
        }

    # 7. faithfulness
    metrics["faithfulness"] = faithfulness_report(predict_fn, c_pred, weights, concept_names=names)

    # 8. stability -- needs concept vectors re-inferred under input perturbation
    perts = _perturbation_names(predictions)
    if perts:
        runs = [_concept_columns(predictions, names, f"__{p}") for p in perts]
        metrics["stability"] = stability_report(
            c_pred, runs, predict_fn=predict_fn, perturbation_names=perts
        )
    else:
        metrics["stability"] = {
            "status": "pending",
            "reason": "no perturbed concept columns in predictions.parquet (needs image access at eval)",
        }

    return metrics
