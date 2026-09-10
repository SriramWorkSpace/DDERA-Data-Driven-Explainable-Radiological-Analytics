"""App component logic (pure functions). The Streamlit pages are thin glue over these.

The load-bearing test: :func:`components.explain.intervene` reports the concept-intervention
identity ``delta logit == sum_j w_j * delta c_j`` exactly -- that identity is the whole
Explainability Lab.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from components.explain import contribution_table, intervene  # noqa: E402
from components.figures import concept_probability_bars, contribution_waterfall  # noqa: E402
from components.runs import (  # noqa: E402
    available_runs,
    latest_run,
    load_reasoner,
    run_is_synthetic,
)
from ddera.xai.intervention import LinearReasoner  # noqa: E402

K = 12


@pytest.fixture
def reasoner() -> LinearReasoner:
    rng = np.random.default_rng(0)
    return LinearReasoner(rng.normal(size=K), -0.4, [f"c{j}" for j in range(K)])


class TestInterventionIdentity:
    def test_single_edit_matches_closed_form_exactly(self, reasoner):
        base = np.full(K, 0.3)
        result = intervene(reasoner, base, {2: 0.9})
        assert result["exact"]
        assert result["logit_delta"] == pytest.approx(reasoner.weights[2] * (0.9 - 0.3), abs=1e-12)

    def test_multi_edit_matches_closed_form(self, reasoner):
        base = np.linspace(0.1, 0.9, K)
        edits = {0: 1.0, 5: 0.0, 11: 0.5}
        result = intervene(reasoner, base, edits)
        assert result["residual"] < 1e-9
        assert result["exact"]

    def test_no_edit_is_a_no_op(self, reasoner):
        base = np.full(K, 0.42)
        result = intervene(reasoner, base, {})
        assert result["logit_delta"] == pytest.approx(0.0, abs=1e-12)
        assert result["base_prob"] == pytest.approx(result["new_prob"])

    def test_contribution_table_sums_to_the_logit(self, reasoner):
        c = np.random.default_rng(1).random(K)
        table = contribution_table(reasoner, c, [f"c{j}" for j in range(K)])
        total = sum(row["contribution"] for row in table)  # includes the (bias) row
        assert total == pytest.approx(float(reasoner.predict_logit(c[None])[0]), abs=1e-9)


class TestFigures:
    def test_bars_and_waterfall_build(self, reasoner):
        c = np.random.default_rng(2).random(K)
        for fig in (
            concept_probability_bars(c, [f"c{j}" for j in range(K)]),
            contribution_waterfall(reasoner, c, [f"c{j}" for j in range(K)]),
        ):
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestRunLoading:
    def test_no_runs_directory_is_empty_not_an_error(self, tmp_path):
        assert available_runs(tmp_path) == []
        assert latest_run(tmp_path) is None

    def test_reads_a_written_run_and_rebuilds_the_reasoner(self, tmp_path):
        import pandas as pd

        from ddera.reporting.runs import log_run

        log_run(
            tmp_path / "cbm_sequential_20260101-000000",
            config={"model": {"variant": "m2_sequential"}, "_synthetic": True},
            metrics={"split": "val", "predictive": {"auroc": 0.6}, "calibration": {"ece": 0.1}},
            predictions=pd.DataFrame(
                {
                    "row": [0, 1],
                    "target": [0, 1],
                    "target_prob": [0.3, 0.7],
                    **{f"concept_prob__c{j}": [0.2, 0.8] for j in range(K)},
                }
            ),
            concept_weights={
                "weights": [0.1] * K,
                "bias": 0.0,
                "concept_names": [f"c{j}" for j in range(K)],
            },
            partial=True,
            index_csv=tmp_path / "runs_index.csv",
        )
        runs = available_runs(tmp_path)
        assert len(runs) == 1
        assert run_is_synthetic(runs[0])
        reasoner, names = load_reasoner(runs[0])
        assert len(names) == K
        assert reasoner.n_concepts == K
