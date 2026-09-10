"""Training loop + evaluator + the `python -m ddera.train` entrypoint.

Built on the ``synthetic_cached`` fixture (a random-init DenseNet feature cache over a
synthetic tree). Nothing here is a result -- these check the machinery runs, the loss goes
down, and a partial run is written with the right shape.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ddera.config import ExperimentConfig, ModelConfig
from ddera.data.dataset import CachedFeatureDataset
from ddera.models.cbm import build_model
from ddera.reporting.runs import load_run
from ddera.seed import set_seed
from ddera.train.evaluate import evaluate_model
from ddera.train.loop import train_model

DEVICE = torch.device("cpu")


def _cfg(**over) -> ExperimentConfig:
    base = ExperimentConfig(
        name="test_m2",
        model=ModelConfig(
            variant="m2_sequential",
            encoder=dataclasses.replace(ModelConfig().encoder, weights=None),
        ),
        epochs=8,
        batch_size=32,
        lr=5e-2,
        early_stop_patience=20,
    )
    return dataclasses.replace(base, **over)


def _loaders(cache_dir, splits, spec, batch_size=32):
    train = CachedFeatureDataset(cache_dir, "train", splits, spec)
    val = CachedFeatureDataset(cache_dir, "val", splits, spec)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        train,
        val,
    )


class _SignalDataset(torch.utils.data.Dataset):
    """Wraps make_synthetic_cbm as CBM batches: features linearly encode the concepts, and
    the concepts linearly drive the target -- so a working loop must reduce the loss.
    """

    def __init__(self, n_patients: int = 400, n_concepts: int = 12) -> None:
        from ddera.data.synthetic import make_synthetic_cbm

        self.s = make_synthetic_cbm(
            n_patients=n_patients,
            studies_per_patient=1,
            n_concepts=n_concepts,
            n_features=64,
            concept_noise=0.3,
            seed=0,
        )
        self.k = n_concepts

    def __len__(self) -> int:
        return self.s.n_samples

    def __getitem__(self, i: int) -> dict:
        return {
            "features": torch.tensor(self.s.features[i], dtype=torch.float32),
            "concepts": torch.tensor(self.s.concepts_true[i], dtype=torch.float32),
            "concept_mask": torch.ones(self.k),
            "target": torch.tensor(float(self.s.y[i])),
        }


def _signal_model(k: int = 12):
    from ddera.models.cbm import ConceptBottleneckModel
    from ddera.models.concept_head import ConceptHead
    from ddera.models.reasoner import LinearReasoner

    return ConceptBottleneckModel(
        ConceptHead(64, k), LinearReasoner(k), concepts_from="predicted", detach_reasoner_input=True
    )


class TestTrainLoop:
    def test_fast_dev_run_completes(self, synthetic_cached):
        cache_dir, splits, spec = synthetic_cached
        tl, vl, _, _ = _loaders(cache_dir, splits, spec)
        set_seed(0)
        model = build_model(_cfg().model)
        history = train_model(
            model, tl, vl, _cfg(), device=DEVICE, regime="sequential", fast_dev_run=True
        )
        assert len(history.epochs) == 1

    def test_sequential_fit_reduces_loss_when_the_concepts_carry_signal(self):
        ds = _SignalDataset(n_patients=400)
        n = len(ds)
        tr = torch.utils.data.Subset(ds, range(int(0.8 * n)))
        va = torch.utils.data.Subset(ds, range(int(0.8 * n), n))
        tl = DataLoader(tr, batch_size=64, shuffle=True)
        vl = DataLoader(va, batch_size=64, shuffle=False)

        set_seed(0)
        model = _signal_model()
        history = train_model(
            model, tl, vl, _cfg(epochs=25, lr=1e-2), device=DEVICE, regime="sequential"
        )
        first = history.epochs[0].val_target_loss
        best = min(e.val_target_loss for e in history.epochs)
        assert best < first - 1e-3
        assert min(e.val_concept_loss for e in history.epochs) < history.epochs[0].val_concept_loss

    def test_early_stopping_can_trigger(self, synthetic_cached):
        cache_dir, splits, spec = synthetic_cached
        tl, vl, _, _ = _loaders(cache_dir, splits, spec)
        set_seed(0)
        model = build_model(_cfg().model)
        history = train_model(
            model,
            tl,
            vl,
            _cfg(epochs=50, early_stop_patience=2),
            device=DEVICE,
            regime="sequential",
        )
        assert history.stopped_early or len(history.epochs) == 50


class TestEvaluator:
    def test_metrics_and_predictions_shape_without_perturbations(self, synthetic_cached):
        cache_dir, splits, spec = synthetic_cached
        _, vl, _, val = _loaders(cache_dir, splits, spec)
        set_seed(0)
        model = build_model(_cfg().model)
        preds, metrics, weights = evaluate_model(model, vl, spec=spec, device=DEVICE, split="val")

        # No perturbation loaders and no baseline -> stability pending, completeness partial,
        # so the run stays partial. Everything else is real.
        assert metrics["status"] == "partial"
        assert {
            "predictive",
            "calibration",
            "concept_quality",
            "faithfulness",
            "intervention",
            "leakage",
        } <= set(metrics["families_present"])
        assert metrics["stability"]["status"] == "pending"
        assert metrics["completeness"]["partial_curve"] is True
        assert set(metrics["intervention"]) == {"random", "uncertainty", "weight", "oracle"}
        assert len(weights["weights"]) == 12
        assert len(preds) == len(val)
        assert "target_prob" in preds.columns
        assert any(c.startswith("concept_prob__") for c in preds.columns)

    def test_perturbation_loaders_and_baseline_make_the_run_complete(self, synthetic_cached):
        cache_dir, splits, spec = synthetic_cached
        _, vl, _, _ = _loaders(cache_dir, splits, spec)
        # Re-use the same val loader as stand-in "perturbations" -- shape is what matters here.
        pert = {k: _loaders(cache_dir, splits, spec)[1] for k in ("rotate", "noise")}
        set_seed(0)

        b0 = build_model(ModelConfig(variant="b0", encoder=_cfg().model.encoder))
        b0_preds, _, _ = evaluate_model(b0, vl, spec=spec, device=DEVICE, split="val")

        model = build_model(_cfg().model)
        _, metrics, _ = evaluate_model(
            model,
            vl,
            spec=spec,
            device=DEVICE,
            split="val",
            perturbation_loaders=pert,
            baseline_predictions=b0_preds,
        )
        assert metrics["stability"].get("status") != "pending"
        assert "aggregate" in metrics["stability"]
        assert metrics["completeness"].get("partial_curve") is not True
        assert metrics["status"] == "complete"

    def test_faithfulness_is_near_perfect_for_the_linear_reasoner(self, synthetic_cached):
        cache_dir, splits, spec = synthetic_cached
        _, vl, _, _ = _loaders(cache_dir, splits, spec)
        set_seed(0)
        _, metrics, _ = evaluate_model(
            build_model(_cfg().model), vl, spec=spec, device=DEVICE, split="val"
        )
        # d logit / d c_j must match w_j for a linear reasoner.
        assert metrics["faithfulness"]["correlation"] > 0.99


class TestEntrypoint:
    def test_synthetic_fast_dev_run_writes_a_partial_run(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ddera.train.__main__ import main

        rc = main(
            [
                "--config",
                "configs/experiment/m2_sequential.yaml",
                "--synthetic",
                "--fast-dev-run",
                "--out",
                str(tmp_path / "runs"),
            ]
        )
        assert rc == 0
        run_dirs = list((tmp_path / "runs").glob("cbm_sequential_*"))
        assert len(run_dirs) == 1
        run = load_run(run_dirs[0])
        assert run["metrics"]["status"] == "partial"
        assert run["metrics"]["synthetic"] is True
        assert (tmp_path / "runs" / "runs_index.csv").exists()

    def test_real_config_without_data_locations_exits_cleanly(self, tmp_path, capsys):
        from ddera.train.__main__ import main

        try:
            main(["--config", "configs/experiment/m2_sequential.yaml", "--out", str(tmp_path)])
        except SystemExit as exc:
            assert "splits_parquet" in str(exc) or "--synthetic" in str(exc)
        else:
            raise AssertionError("expected SystemExit when no data locations are configured")
