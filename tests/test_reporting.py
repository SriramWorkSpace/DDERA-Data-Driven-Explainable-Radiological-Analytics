"""Reporting theme + progress-figure functions.

These check that each plot builds a real Figure from real (synthetic-pipeline) artifacts and
writes a non-empty PNG. They assert nothing about figure *content* -- these are progress
views, not results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from ddera.data.chexpert import load_manifest  # noqa: E402
from ddera.data.splits import load_splits  # noqa: E402
from ddera.reporting import plots  # noqa: E402
from ddera.reporting.theme import apply_theme, mark_synthetic, save_figure  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_theme():
    apply_theme()
    yield
    plt.close("all")


class TestTheme:
    def test_mark_synthetic_adds_one_text_artist(self):
        fig, _ = plt.subplots()
        before = len(fig.texts)
        mark_synthetic(fig)
        assert len(fig.texts) == before + 1

    def test_save_figure_writes_a_nonempty_png(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = save_figure(fig, tmp_path / "sub" / "f.png", synthetic=True)
        assert out.exists() and out.stat().st_size > 0


class TestPlots:
    def test_every_plot_returns_a_figure_and_saves(self, synthetic_processed, tmp_path):
        out, _, spec = synthetic_processed
        manifest, _ = load_manifest(out / "manifest.parquet")
        splits_df, _ = load_splits(out / "splits.parquet")
        observations = [spec.target, *spec.concepts]

        figures = [
            plots.plot_split_summary(splits_df),
            plots.plot_target_prevalence_by_split(splits_df),
            plots.plot_label_distribution(manifest, observations),
            plots.plot_concept_cooccurrence(manifest, spec.concepts),
            plots.plot_mask_coverage(manifest, spec),
        ]
        for i, fig in enumerate(figures):
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) >= 1
            path = save_figure(fig, tmp_path / f"fig_{i}.png", synthetic=True)
            assert path.stat().st_size > 0


class TestVisualizeDatasetCLI:
    def test_synthetic_demo_writes_five_figures(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.visualize_dataset import main

        out = tmp_path / "figs"
        assert main(["--synthetic-demo", "--out", str(out)]) == 0
        assert len(list(out.glob("*.png"))) == 5

    def test_missing_artifacts_exit_2(self, tmp_path, capsys):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.visualize_dataset import main

        rc = main(["--processed", str(tmp_path / "nope"), "--out", str(tmp_path / "figs")])
        assert rc == 2
        assert "not found" in capsys.readouterr().err
