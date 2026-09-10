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


class TestPreprocessingPlots:
    def test_augmentation_and_sample_grids(self, synthetic_processed, tmp_path):
        import numpy as np

        from ddera.config import EncoderConfig
        from ddera.data.dataset import CheXpertImageDataset
        from ddera.data.transforms import build_eval_transform, build_train_transform

        out, root, spec = synthetic_processed
        ds = CheXpertImageDataset(
            out / "splits.parquet",
            "train",
            spec,
            data_root=root,
            transform=build_eval_transform(),
        )
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, :32] = 200
        fig_a = plots.plot_augmentation_grid(img, build_train_transform(), n=4)
        fig_s = plots.plot_preprocessed_samples(ds, n=4, encoder=EncoderConfig())
        for fig in (fig_a, fig_s):
            assert isinstance(fig, plt.Figure)
            assert save_figure(fig, tmp_path / f"{id(fig)}.png").stat().st_size > 0

    def test_feature_space_and_cache_sanity(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(0)
        feats = rng.normal(size=(60, 32))
        colours = (feats[:, 0] > 0).astype(int)
        fig_fs = plots.plot_feature_space(feats, colours)
        fig_cs = plots.plot_feature_cache_sanity(feats.astype(np.float16), feats)
        for fig in (fig_fs, fig_cs):
            assert isinstance(fig, plt.Figure)
            assert save_figure(fig, tmp_path / f"{id(fig)}.png").stat().st_size > 0


class TestVisualizeCLIs:
    def test_visualize_dataset_synthetic_demo(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.visualize_dataset import main

        out = tmp_path / "figs"
        assert main(["--synthetic-demo", "--out", str(out)]) == 0
        assert len(list(out.glob("*.png"))) == 5

    def test_visualize_dataset_missing_artifacts_exit_2(self, tmp_path, capsys):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.visualize_dataset import main

        rc = main(["--processed", str(tmp_path / "nope"), "--out", str(tmp_path / "figs")])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_visualize_preprocessing_synthetic_demo(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.visualize_preprocessing import main

        out = tmp_path / "prep"
        assert main(["--synthetic-demo", "--out", str(out)]) == 0
        # augmentation grid, preprocessed samples, cache sanity, feature space, probe
        assert len(list(out.glob("*.png"))) >= 4
