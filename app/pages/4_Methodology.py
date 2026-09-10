"""Methodology page -- the pathway, the evaluation protocol, the limitations."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.disclaimer import render_disclaimer  # noqa: E402

st.set_page_config(page_title="DDERA - Methodology", page_icon="🫁", layout="wide")
st.title("Methodology")
render_disclaimer(st)

st.markdown(
    """
### The ante-hoc pathway

```
Chest X-ray -> DenseNet-121 -> concept head -> concept vector c in [0,1]^12
            -> linear reasoner  logit(p) = b + sum_j w_j c_j -> P(Pneumonia)
```

The reasoner sees **only** the 12-concept vector (Invariant 4). Because it is linear the
logit decomposes exactly into signed per-concept contributions `w_j c_j` -- no attribution
method, no sampling. Correcting a concept moves the prediction by exactly `w_j * delta c_j`;
that is the Explainability Lab.

### The eight evaluation families (ARCHITECTURE section 8)

| # | Family | Answers |
|---|---|---|
| 1 | Predictive performance | Is it accurate? |
| 2 | Calibration | Are its probabilities meaningful? |
| 3 | Concept quality | Can it identify its own concepts? |
| 4 | Intervention curves (TTI) | Does correcting concepts help? |
| 5 | Leakage | Does information bypass the bottleneck? |
| 6 | Completeness | How much do the concepts fail to carry? |
| 7 | Faithfulness | Does d p / d c_j match the claimed w_j? |
| 8 | Stability | Do concepts survive irrelevant perturbation? |

A run is not `complete` until `metrics.json` carries all eight
(`ddera.reporting.runs` enforces this). Phase-3 runs are **partial**: families 5, 6 and 8
arrive with the Phase-5 protocol.

### Model variants

| ID | Model | Reasoner input | Purpose |
|---|---|---|---|
| B0 | Black box | 1024-d features | accuracy ceiling (not ante-hoc) |
| M1 | Independent CBM | ground-truth concepts | max interpretability |
| M2 | Sequential CBM | predicted concepts | the practical default |
| M3 | Joint CBM | predicted concepts | lambda sweep -> trade-off curve (Phase 4) |
| M4 | Hybrid / residual | concepts + residual(k) | completeness curve (Phase 4) |

### Limitations

- CheXpert concept labels are NLP-extracted from radiology reports, not annotated on the
  images (ADR-002). Phase 8 (VinDr-CXR) addresses this with human annotations.
- The primary test set is a patient-disjoint 20% carved from `train`; official CheXpert
  test labels are not public (ADR-005).
- Frontal views only in v1 (ADR-006).
"""
)
