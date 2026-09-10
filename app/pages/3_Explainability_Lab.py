"""Explainability Lab -- the demonstration of the whole thesis.

Pick a run and a sample; see the predicted concept vector, the contribution waterfall
(``w_j c_j``, signed, summing exactly to the logit), then move the sliders and watch the
prediction shift by **exactly** ``sum_j w_j * delta c_j`` in logit space. The identity is
checked live and shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.disclaimer import NO_RUNS_HINT, render_disclaimer, render_state_banner  # noqa: E402
from components.explain import contribution_table, intervene  # noqa: E402
from components.figures import concept_probability_bars, contribution_waterfall  # noqa: E402
from components.runs import (  # noqa: E402
    available_runs,
    load_reasoner,
    run_is_synthetic,
    sample_concepts,
)

st.set_page_config(page_title="DDERA - Explainability Lab", page_icon="🫁", layout="wide")
st.title("Explainability Lab")
render_disclaimer(st)

runs = [r for r in available_runs() if load_reasoner(r) is not None]
if not runs:
    st.info(NO_RUNS_HINT + "\n\n(The Lab needs a concept-bottleneck run; B0 has no reasoner.)")
    st.stop()

by_id = {r["run_id"]: r for r in runs}
run = by_id[st.selectbox("Run", list(by_id))]
render_state_banner(st, synthetic=run_is_synthetic(run))

reasoner, names = load_reasoner(run)
n_rows = len(run["predictions"]) if run.get("predictions") is not None else 0
row = st.number_input("Sample row (from predictions.parquet)", 0, max(n_rows - 1, 0), 0)
base = sample_concepts(run, int(row))
if base is None:
    base = np.full(len(names), 0.5)
    st.caption("No predictions on file for this run; starting from a neutral 0.5 vector.")

st.sidebar.header("Intervene on concepts")
edits: dict[int, float] = {}
for j, name in enumerate(names):
    new = st.sidebar.slider(name, 0.0, 1.0, float(round(base[j], 3)), 0.01, key=f"c{j}")
    if abs(new - base[j]) > 1e-9:
        edits[j] = new

result = intervene(reasoner, base, edits)

m1, m2, m3 = st.columns(3)
m1.metric("P(Pneumonia) - predicted", f"{result['base_prob']:.3f}")
m2.metric(
    "P(Pneumonia) - after intervention",
    f"{result['new_prob']:.3f}",
    delta=f"{result['new_prob'] - result['base_prob']:+.3f}",
)
m3.metric("logit shift", f"{result['logit_delta']:+.4f}")

st.markdown(
    f"**Closed form** `sum_j w_j * delta c_j = {result['expected_delta']:+.4f}` &nbsp; vs &nbsp; "
    f"**measured** `delta logit = {result['logit_delta']:+.4f}` &nbsp; -> residual "
    f"`{result['residual']:.2e}`"
)
(st.success if result["exact"] else st.error)(
    "Identity holds: the explanation *is* the mechanism."
    if result["exact"]
    else "Identity broken - the displayed contributions do not match the prediction."
)

col_l, col_r = st.columns(2)
with col_l:
    st.pyplot(
        concept_probability_bars(base, names, edited=result["edited_concepts"] if edits else None)
    )
with col_r:
    st.pyplot(contribution_waterfall(reasoner, result["edited_concepts"], names))

st.subheader("Contribution table (edited vector)")
st.dataframe(
    pd.DataFrame(contribution_table(reasoner, result["edited_concepts"], names)),
    use_container_width=True,
)
