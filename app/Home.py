"""DDERA dashboard -- Home.

    streamlit run app/Home.py

Reads run artifacts from ``experiments/runs/`` (never hardcoded numbers). Until the CheXpert
download exists, the demo run is synthetic and every page says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from components.disclaimer import NO_RUNS_HINT, render_disclaimer, render_state_banner  # noqa: E402
from components.runs import available_runs, run_is_synthetic  # noqa: E402

st.set_page_config(page_title="DDERA", page_icon="🫁", layout="wide")

st.title("DDERA - Data-Driven Explainable Radiological Analytics")
st.markdown(
    "An **ante-hoc** concept-bottleneck model: `image -> DenseNet-121 -> 12 clinical "
    "concepts -> linear reasoner -> Pneumonia`. The explanation lies *on* the prediction "
    "path, not beside it. Post-hoc methods (SHAP / LIME / Grad-CAM) are comparison "
    "baselines only."
)
render_disclaimer(st)

runs = available_runs()
if not runs:
    st.info(NO_RUNS_HINT)
    st.stop()

render_state_banner(st, synthetic=any(run_is_synthetic(r) for r in runs))

st.subheader("Runs")
rows = []
for r in runs:
    m = r["metrics"]
    predictive = m.get("predictive", {})
    rows.append(
        {
            "run_id": r["run_id"],
            "variant": r["config"].get("model", {}).get("variant", ""),
            "status": m.get("status"),
            "split": m.get("split"),
            "AUROC": predictive.get("auroc"),
            "AUROC 95% CI": predictive.get("auroc_ci"),
            "synthetic": run_is_synthetic(r),
        }
    )
st.dataframe(rows, use_container_width=True)

st.caption(
    "Pages: **Data** (cohort & label distributions) - **Model** (metrics & calibration) - "
    "**Explainability Lab** (live concept intervention) - **Methodology** (the protocol)."
)
