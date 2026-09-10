"""Model page -- predictive metrics, calibration, per-concept quality, from run artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.disclaimer import NO_RUNS_HINT, render_disclaimer, render_state_banner  # noqa: E402
from components.figures import roc_curve_figure  # noqa: E402
from components.runs import available_runs, run_is_synthetic  # noqa: E402

st.set_page_config(page_title="DDERA - Model", page_icon="🫁", layout="wide")
st.title("Model")
render_disclaimer(st)

runs = available_runs()
if not runs:
    st.info(NO_RUNS_HINT)
    st.stop()

by_id = {r["run_id"]: r for r in runs}
run = by_id[st.selectbox("Run", list(by_id))]
render_state_banner(st, synthetic=run_is_synthetic(run))

metrics = run["metrics"]
predictive = metrics.get("predictive", {})
calibration = metrics.get("calibration", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUROC", f"{predictive.get('auroc', float('nan')):.3f}")
c2.metric("AUPRC", f"{predictive.get('auprc', float('nan')):.3f}")
c3.metric("Brier", f"{calibration.get('brier', float('nan')):.3f}")
c4.metric("ECE", f"{calibration.get('ece', float('nan')):.3f}")
if predictive.get("auroc_ci"):
    lo, hi = predictive["auroc_ci"]
    st.caption(f"AUROC 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]  (status: {metrics.get('status')})")

preds = run.get("predictions")
if preds is not None and "target_prob" in preds:
    st.pyplot(roc_curve_figure(preds["target"].to_numpy(), preds["target_prob"].to_numpy()))

cq = metrics.get("concept_quality")
if isinstance(cq, dict) and "per_concept" in cq:
    st.subheader("Per-concept quality")
    st.dataframe(
        pd.DataFrame(cq["per_concept"]).T.reset_index(names="concept"),
        use_container_width=True,
    )

pending = [
    k
    for k in ("leakage", "completeness", "stability")
    if metrics.get(k, {}).get("status") == "pending"
]
if pending:
    st.info(f"Pending metric families (Phase 5): {', '.join(pending)}")
