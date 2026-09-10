"""The trainable concept-bottleneck model -- forward-graph contract (Invariants 3/4).

The load-bearing tests here: the target logit depends on the encoder representation ONLY
through the k-wide concept vector, and the linear reasoner's contributions sum exactly to
the logit. ``tests/test_cbm_math.py`` covers the numpy analysis reasoner; this covers the
torch model it is bridged from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ddera.config import ModelConfig
from ddera.models.blackbox import BlackBoxModel
from ddera.models.cbm import build_model
from ddera.models.reasoner import LinearReasoner
from ddera.seed import set_seed
from ddera.train.losses import cbm_loss, masked_bce_with_logits

K = 12
D = 1024


def _feature_batch(n: int = 8, *, grad: bool = False) -> dict:
    set_seed(0)
    return {
        "features": torch.randn(n, D, requires_grad=grad),
        "concepts": torch.randint(0, 2, (n, K)).float(),
        "concept_mask": torch.ones(n, K),
        "target": torch.randint(0, 2, (n,)).float(),
    }


class TestForward:
    def test_m2_output_keys_and_shapes(self):
        set_seed(0)
        model = build_model(ModelConfig(variant="m2_sequential"))
        out = model(_feature_batch(8))
        assert set(out) == {"concept_logits", "concept_probs", "target_logit"}
        assert out["concept_logits"].shape == (8, K)
        assert out["target_logit"].shape == (8,)
        assert torch.all((out["concept_probs"] >= 0) & (out["concept_probs"] <= 1))

    def test_m4_is_not_implemented_yet(self):
        with pytest.raises(NotImplementedError, match="Phase 4"):
            build_model(ModelConfig(variant="m4_hybrid"))


class TestInvariant34ForwardGraph:
    def test_reasoner_input_width_is_exactly_n_concepts(self):
        model = build_model(ModelConfig(variant="m2_sequential"))
        assert model.reasoner.linear.in_features == K

    def test_target_logit_is_invariant_to_feature_changes_that_leave_concepts_fixed(self):
        """The ante-hoc guarantee: move the representation within the concept head's null
        space and the prediction does not budge, because nothing bypasses the bottleneck.
        """
        set_seed(0)
        model = build_model(ModelConfig(variant="m2_sequential")).eval()
        batch = _feature_batch(4)

        weight = model.concept_head.linear.weight.detach()  # (K, D)
        # A direction in feature space that the concept head maps to zero.
        _, _, vh = torch.linalg.svd(weight, full_matrices=True)
        null_dir = vh[K:][0]  # first basis vector of the (D-K)-dim null space
        assert torch.allclose(weight @ null_dir, torch.zeros(K), atol=1e-4)

        base = model(batch)["target_logit"]
        moved = dict(batch)
        moved["features"] = batch["features"] + 25.0 * null_dir
        after = model(moved)["target_logit"]
        assert torch.allclose(base, after, atol=1e-4)

    def test_gradient_from_target_reaches_features_only_via_the_concept_head(self):
        set_seed(0)
        model = build_model(ModelConfig(variant="m3_joint"))  # joint => grad flows to the head
        batch = _feature_batch(4, grad=True)
        model(batch)["target_logit"].sum().backward()
        # The concept head is on the path...
        assert model.concept_head.linear.weight.grad is not None
        # ...and the only parameter tensor with a gradient besides the reasoner.
        grad_params = {n for n, p in model.named_parameters() if p.grad is not None}
        assert grad_params <= {
            "concept_head.linear.weight",
            "concept_head.linear.bias",
            "reasoner.linear.weight",
            "reasoner.linear.bias",
        }

    def test_sequential_regime_detaches_the_concept_head_from_the_target_loss(self):
        set_seed(0)
        model = build_model(ModelConfig(variant="m2_sequential"))
        batch = _feature_batch(4)
        out = model(batch)
        # Only the target term, back-propagated: the detached reasoner input means the head
        # receives no gradient from it.
        from ddera.train.losses import target_bce

        target_bce(out["target_logit"], batch["target"]).backward()
        assert (
            model.concept_head.linear.weight.grad is None
            or torch.count_nonzero(model.concept_head.linear.weight.grad) == 0
        )
        assert model.reasoner.linear.weight.grad is not None


class TestExactDecomposition:
    def test_contributions_plus_bias_equal_the_logit(self):
        set_seed(1)
        reasoner = LinearReasoner(K)
        concepts = torch.rand(16, K)
        recon = reasoner.contributions(concepts).sum(dim=1) + reasoner.bias_value
        assert torch.allclose(recon, reasoner(concepts), atol=1e-5)

    def test_bridge_to_numpy_reasoner_matches(self):
        set_seed(2)
        reasoner = LinearReasoner(K)
        concepts = torch.rand(10, K)
        np_reasoner = reasoner.to_numpy_reasoner([f"c{j}" for j in range(K)])
        np.testing.assert_allclose(
            np_reasoner.predict_logit(concepts.numpy()),
            reasoner(concepts).detach().numpy(),
            atol=1e-5,
        )


class TestMaskedBCE:
    def test_masked_elements_do_not_affect_the_loss(self):
        set_seed(0)
        logits = torch.randn(4, K, requires_grad=True)
        targets = torch.randint(0, 2, (4, K)).float()
        mask = torch.ones(4, K)
        mask[:, 0] = 0.0  # hide concept 0 entirely

        full = masked_bce_with_logits(logits, targets, mask)
        # Changing the hidden column's targets must not move the loss.
        targets2 = targets.clone()
        targets2[:, 0] = 1.0 - targets2[:, 0]
        assert torch.allclose(full, masked_bce_with_logits(logits, targets2, mask))

    def test_masked_column_gets_no_gradient(self):
        logits = torch.randn(4, K, requires_grad=True)
        targets = torch.randint(0, 2, (4, K)).float()
        mask = torch.ones(4, K)
        mask[:, 3] = 0.0
        masked_bce_with_logits(logits, targets, mask).backward()
        assert torch.count_nonzero(logits.grad[:, 3]) == 0
        assert torch.count_nonzero(logits.grad[:, 1]) > 0


class TestBlackBoxHasADirectPath:
    def test_b0_gradient_flows_features_to_target_without_a_bottleneck(self):
        set_seed(0)
        model = build_model(ModelConfig(variant="b0"))
        assert isinstance(model, BlackBoxModel)
        batch = _feature_batch(4, grad=True)
        model(batch)["target_logit"].sum().backward()
        assert batch["features"].grad is not None
        assert torch.count_nonzero(batch["features"].grad) > 0


class TestLossRegimes:
    def test_joint_weights_the_concept_term(self):
        set_seed(0)
        model = build_model(ModelConfig(variant="m3_joint"))
        batch = _feature_batch(8)
        out = model(batch)
        a = cbm_loss(out, batch, regime="joint", concept_weight=0.0)
        b = cbm_loss(out, batch, regime="joint", concept_weight=5.0)
        assert not torch.allclose(a["loss"], b["loss"])
        assert torch.allclose(a["loss"], b["target_loss"])  # lambda=0 => target term only

    def test_bad_regime_rejected(self):
        model = build_model(ModelConfig(variant="m2_sequential"))
        batch = _feature_batch(4)
        with pytest.raises(ValueError, match="regime must be one of"):
            cbm_loss(model(batch), batch, regime="bogus")


def test_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown model variant"):
        ModelConfig(variant="transformer")
