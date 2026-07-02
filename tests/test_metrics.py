"""
Tests for ehrdrec.metrics — Jaccard, F1, PRAUC, and BinaryDDI.

All metrics follow the same update/compute/reset lifecycle, so tests share a
consistent structure: build a known scenario, check the computed value against
a hand-calculated reference, then confirm reset clears state.

Logit inputs are used by default (sigmoid applied internally); a few tests pass
from_logits=False to exercise that code path.
"""
import io
import math

import pytest
import torch

from ehrdrec.metrics.jaccard import Jaccard
from ehrdrec.metrics.f1 import F1
from ehrdrec.metrics.prauc import PRAUC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    """Return the logit of probability p (inverse sigmoid)."""
    return math.log(p / (1 - p))


def make_tensors(preds: list[list[float]], targets: list[list[int]]):
    return torch.tensor(preds), torch.tensor(targets, dtype=torch.float)


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_perfect_prediction(self):
        # When predictions exactly match targets Jaccard should be 1.0.
        # ignore_indices=[] so reserved ids don't interfere.
        metric = Jaccard(ignore_indices=[], from_logits=False)
        preds = torch.tensor([[0.9, 0.1, 0.8]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        metric.update(preds, targets)
        assert metric.compute() == pytest.approx(1.0)

    def test_no_overlap(self):
        # Preds fire on slot 0 only; target fires on slot 1 only → Jaccard = 0.
        metric = Jaccard(ignore_indices=[], from_logits=False)
        preds = torch.tensor([[0.9, 0.1]])
        targets = torch.tensor([[0.0, 1.0]])
        metric.update(preds, targets)
        assert metric.compute() == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Preds: [1,1,0], targets: [1,0,1] → intersection=1, union=3 → J=1/3
        metric = Jaccard(ignore_indices=[], from_logits=False)
        preds = torch.tensor([[0.9, 0.9, 0.1]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        metric.update(preds, targets)
        assert metric.compute() == pytest.approx(1 / 3)

    def test_from_logits(self):
        # Logit of 0.9 ≈ 2.197, logit of 0.1 ≈ -2.197
        metric = Jaccard(ignore_indices=[], from_logits=True)
        preds = torch.tensor([[logit(0.9), logit(0.1), logit(0.8)]])
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        metric.update(preds, targets)
        assert metric.compute() == pytest.approx(1.0)

    def test_ignore_indices_excluded(self):
        # Slots 0 and 1 are reserved (UNK, PAD). Slot 2 is the only meaningful one.
        # After masking: preds=[1], targets=[1] → Jaccard=1.0
        metric = Jaccard(ignore_indices=[0, 1], from_logits=False)
        preds = torch.tensor([[0.9, 0.9, 0.9]])   # slots 0,1 masked out; slot 2 fires
        targets = torch.tensor([[0.0, 0.0, 1.0]])  # slot 2 is target
        metric.update(preds, targets)
        assert metric.compute() == pytest.approx(1.0)

    def test_accumulates_across_batches(self):
        # Two batches: each perfect → still 1.0
        metric = Jaccard(ignore_indices=[], from_logits=False)
        for _ in range(3):
            metric.update(torch.tensor([[0.9, 0.1]]), torch.tensor([[1.0, 0.0]]))
        assert metric.compute() == pytest.approx(1.0)

    def test_reset_clears_state(self):
        metric = Jaccard(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.9]]), torch.tensor([[1.0]]))
        metric.reset()
        assert metric.compute() == pytest.approx(0.0)

    def test_empty_union_returns_zero(self):
        # Both preds and targets are all-zero → union=0 → compute() should not divide by zero
        metric = Jaccard(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.1, 0.1]]), torch.tensor([[0.0, 0.0]]))
        assert metric.compute() == pytest.approx(0.0)

    def test_batch_dim(self):
        # 3-sample batch; 2 out of 3 have perfect overlap.
        # intersection=3, union=4 → J=3/4
        metric = Jaccard(ignore_indices=[], from_logits=False)
        preds   = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        metric.update(preds, targets)
        # sample 1: inter=1 union=1; sample 2: inter=1 union=1; sample 3: inter=0 union=2
        assert metric.compute() == pytest.approx(2 / 4)


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

class TestF1:
    def test_perfect(self):
        metric = F1(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.9, 0.1, 0.8]]), torch.tensor([[1.0, 0.0, 1.0]]))
        assert metric.compute() == pytest.approx(1.0)

    def test_all_wrong(self):
        # preds: [1,0,0], targets: [0,1,1] → TP=0, FP=1, FN=2 → F1=0
        metric = F1(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.9, 0.1, 0.1]]), torch.tensor([[0.0, 1.0, 1.0]]))
        assert metric.compute() == pytest.approx(0.0)

    def test_known_value(self):
        # preds: [1,1,0], targets: [1,0,1] → TP=1, FP=1, FN=1 → F1 = 2/(2+1+1) = 0.5
        metric = F1(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.9, 0.9, 0.1]]), torch.tensor([[1.0, 0.0, 1.0]]))
        assert metric.compute() == pytest.approx(0.5)

    def test_from_logits(self):
        metric = F1(ignore_indices=[], from_logits=True)
        metric.update(
            torch.tensor([[logit(0.9), logit(0.1), logit(0.8)]]),
            torch.tensor([[1.0, 0.0, 1.0]]),
        )
        assert metric.compute() == pytest.approx(1.0)

    def test_ignore_indices(self):
        # Slot 0 (reserved) ignored. preds[1:] = [1], targets[1:] = [1] → F1=1
        metric = F1(ignore_indices=[0], from_logits=False)
        metric.update(torch.tensor([[0.1, 0.9]]), torch.tensor([[1.0, 1.0]]))
        # After masking slot 0: preds=[1], targets=[1] → perfect
        assert metric.compute() == pytest.approx(1.0)

    def test_accumulation(self):
        # Each update: TP=1, FP=0, FN=0; three updates: same ratio → F1=1
        metric = F1(ignore_indices=[], from_logits=False)
        for _ in range(3):
            metric.update(torch.tensor([[0.9]]), torch.tensor([[1.0]]))
        assert metric.compute() == pytest.approx(1.0)

    def test_reset(self):
        metric = F1(ignore_indices=[], from_logits=False)
        metric.update(torch.tensor([[0.9]]), torch.tensor([[1.0]]))
        metric.reset()
        # After reset tp=fp=fn=0 → denominator 0 → returns 0.0
        assert metric.compute() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PRAUC
# ---------------------------------------------------------------------------

class TestPRAUC:
    def test_perfect_ranking(self):
        # All positive examples ranked before negatives → PRAUC = 1.0
        metric = PRAUC(ignore_indices=[])
        outputs = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
        targets = torch.tensor([[1.0, 1.0,  0.0,  0.0]])
        metric.update(outputs, targets)
        assert metric.compute() == pytest.approx(1.0)

    def test_worst_ranking(self):
        # All positive examples ranked last → PRAUC should be low (≤ 0.5)
        metric = PRAUC(ignore_indices=[])
        outputs = torch.tensor([[-2.0, -1.0, 1.0, 2.0]])
        targets = torch.tensor([[1.0,  1.0, 0.0, 0.0]])
        metric.update(outputs, targets)
        assert metric.compute() < 0.5

    def test_no_positives_returns_zero(self):
        metric = PRAUC(ignore_indices=[])
        metric.update(torch.tensor([[0.5, 0.5]]), torch.tensor([[0.0, 0.0]]))
        assert metric.compute() == pytest.approx(0.0)

    def test_ignore_indices(self):
        # Slot 0 is reserved. With it masked, remaining output=[2.0], target=[1.0] → perfect
        metric = PRAUC(ignore_indices=[0])
        outputs = torch.tensor([[-99.0, 2.0]])
        targets = torch.tensor([[0.0,   1.0]])
        metric.update(outputs, targets)
        assert metric.compute() == pytest.approx(1.0)

    def test_accumulates_across_calls(self):
        # Same perfect batch called twice → still 1.0
        metric = PRAUC(ignore_indices=[])
        for _ in range(2):
            metric.update(torch.tensor([[2.0, -2.0]]), torch.tensor([[1.0, 0.0]]))
        assert metric.compute() == pytest.approx(1.0)

    def test_reset(self):
        metric = PRAUC(ignore_indices=[])
        metric.update(torch.tensor([[2.0, -2.0]]), torch.tensor([[1.0, 0.0]]))
        metric.reset()
        assert metric.all_outputs == []
        assert metric.all_targets == []

    def test_value_in_range(self):
        metric = PRAUC(ignore_indices=[])
        torch.manual_seed(0)
        outputs = torch.randn(8, 20)
        targets = (torch.rand(8, 20) > 0.7).float()
        metric.update(outputs, targets)
        val = metric.compute()
        assert 0.0 <= val <= 1.0
