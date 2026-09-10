"""Data page -- cohort, label and concept distributions from the Phase-1 artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.disclaimer import render_disclaimer  # noqa: E402
from ddera.config import PROCESSED_DATA_ROOT, ConceptSpec  # noqa: E402
from ddera.data.chexpert import load_manifest  # noqa: E402
from ddera.data.splits import load_splits  # noqa: E402
from ddera.reporting import plots  # noqa: E402
from ddera.reporting.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="DDERA - Data", page_icon="🫁", layout="wide")
st.title("Data")
render_disclaimer(st)
apply_theme()

processed = Path(st.text_input("Processed-data directory", str(PROCESSED_DATA_ROOT)))
manifest_path = processed / "manifest.parquet"
splits_path = processed / "splits.parquet"

if not manifest_path.exists() or not splits_path.exists():
    st.info(
        f"No `manifest.parquet` / `splits.parquet` under `{processed}`. Run "
        "`python scripts/get_data.py --dest data/chexpert` first, or "
        "`python scripts/visualize_dataset.py --synthetic-demo` for a labelled synthetic preview."
    )
    st.stop()

spec = ConceptSpec.from_yaml("configs/concepts/chexpert_v1.yaml")
manifest, _ = load_manifest(manifest_path)
splits_df, _ = load_splits(splits_path)
observations = [spec.target, *spec.concepts]

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plots.plot_split_summary(splits_df))
    st.pyplot(plots.plot_target_prevalence_by_split(splits_df))
with col2:
    st.pyplot(plots.plot_mask_coverage(manifest, spec))

st.pyplot(plots.plot_label_distribution(manifest, observations))
st.pyplot(plots.plot_concept_cooccurrence(manifest, spec.concepts))
