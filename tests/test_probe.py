"""Linear concept probe -- the GATE 2 wiring check.

Uses ``make_synthetic_cbm``: its ``features`` are a linear projection of ``concepts_true``
plus noise, so a working probe recovers the concepts well above chance and a probe on
shuffled features does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddera.data.synthetic import make_synthetic_cbm
from ddera.features.probe import linear_probe_concept_auroc


@pytest.fixture(scope="module")
def split_synthetic():
    s = make_synthetic_cbm(n_patients=200, studies_per_patient=1, n_concepts=8, seed=0)
    n = s.n_samples
    tr = np.arange(n) < int(0.6 * n)
    ev = ~tr
    ones = np.ones_like(s.concepts_true)
    return {
        "feat_tr": s.features[tr],
        "y_tr": s.concepts_true[tr],
        "m_tr": ones[tr],
        "feat_ev": s.features[ev],
        "y_ev": s.concepts_true[ev],
        "m_ev": ones[ev],
        "names": s.concept_names,
    }


def _run(d, feat_tr=None):
    return linear_probe_concept_auroc(
        feat_tr if feat_tr is not None else d["feat_tr"],
        d["y_tr"],
        d["m_tr"],
        d["feat_ev"],
        d["y_ev"],
        d["m_ev"],
        d["names"],
    )


def test_recovers_concepts_well_above_chance(split_synthetic):
    result = _run(split_synthetic)
    assert result["n_concepts_scored"] == 8
    assert result["macro_auroc"] > 0.75


def test_shuffled_features_probe_near_chance(split_synthetic):
    shuffled = split_synthetic["feat_tr"].copy()
    np.random.default_rng(0).shuffle(shuffled)  # break the row alignment
    result = _run(split_synthetic, feat_tr=shuffled)
    assert result["macro_auroc"] < 0.65


def test_masked_rows_are_dropped_per_concept(split_synthetic):
    d = dict(split_synthetic)
    d["m_tr"] = d["m_tr"].copy()
    d["m_tr"][: len(d["m_tr"]) // 2, 0] = 0.0  # hide half of concept 0's training labels
    result = _run(d)
    full = _run(split_synthetic)["per_concept"][d["names"][0]]["n_train"]
    assert result["per_concept"][d["names"][0]]["n_train"] < full


def test_plot_helper_builds_a_figure(split_synthetic):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ddera.reporting.plots import plot_concept_probe_auroc
    from ddera.reporting.theme import apply_theme

    apply_theme()
    fig = plot_concept_probe_auroc(_run(split_synthetic))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
