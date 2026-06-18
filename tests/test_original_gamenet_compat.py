import polars as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.datasets.original_gamenet import (
    OriginalGAMENetDataset,
    collate_original_gamenet,
)
from ehrdrec.training.losses.original_gamenet_loss import OriginalGAMENetLoss
from ehrdrec.training.original_gamenet import OriginalGAMENetTrainer


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "patient_id": [1, 1, 1, 2],
            "time": [0, 1, 2, 0],
            "diagnosis_ids": [[2], [3, 4], [], [1]],
            "procedure_ids": [[0], [1], [2], [3]],
            "medication_multihot": [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ],
        }
    )


class OriginalStyleModel(nn.Module):
    def __init__(self, n_medications: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_medications))

    def forward(self, history):
        logits = self.bias.unsqueeze(0)
        if self.training:
            return logits, logits.sigmoid().mean()
        return logits


class MeanLogitMetric:
    name = "MeanLogit"

    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    def update(self, outputs, targets):
        self.values.append(outputs.mean())

    def compute(self):
        return torch.stack(self.values).mean()


class TestOriginalGAMENetDataset:
    def test_returns_original_visit_history_shape(self):
        ds = OriginalGAMENetDataset(
            _df(),
            target_col="medication_multihot",
            patient_id_col="patient_id",
            time_col="time",
        )

        history, target = ds[1]

        assert history == [
            [[2], [0], [0]],
            [[3, 4], [1], [1, 2]],
        ]
        assert target.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_look_back_limits_history(self):
        ds = OriginalGAMENetDataset(
            _df(),
            target_col="medication_multihot",
            patient_id_col="patient_id",
            time_col="time",
            look_back=1,
        )

        history, _ = ds[2]

        assert len(history) == 2
        assert history[0][0] == [3, 4]
        assert history[1][0] == []

    def test_sparse_medication_history_column(self):
        df = _df().with_columns(
            pl.Series("medication_sparse", [[0], [1, 2], [3], [1]])
        )
        ds = OriginalGAMENetDataset(
            df,
            target_col="medication_multihot",
            medication_history_col="medication_sparse",
            medication_history_is_multihot=False,
            patient_id_col="patient_id",
            time_col="time",
        )

        history, _ = ds[1]

        assert history[1][2] == [1, 2]


class TestCollateOriginalGAMENet:
    def test_collates_histories_without_padding(self):
        ds = OriginalGAMENetDataset(
            _df(),
            target_col="medication_multihot",
            patient_id_col="patient_id",
            time_col="time",
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_original_gamenet)

        histories, targets = next(iter(loader))

        assert isinstance(histories, list)
        assert len(histories) == 2
        assert targets.shape == (2, 4)


class TestOriginalGAMENetLoss:
    def test_adds_ddi_penalty(self):
        loss_fn = OriginalGAMENetLoss(ddi_weight=0.5)
        predictions = torch.zeros(1, 4)
        targets = torch.zeros(1, 4)

        loss = loss_fn(
            predictions,
            targets,
            losses={"ddi_loss": torch.tensor(2.0)},
        )

        assert loss.item() == pytest.approx(nn.BCEWithLogitsLoss()(predictions, targets).item() + 1.0)


class TestOriginalGAMENetTrainer:
    def test_fit_runs_with_original_model_contract(self):
        ds = OriginalGAMENetDataset(
            _df(),
            target_col="medication_multihot",
            patient_id_col="patient_id",
            time_col="time",
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_original_gamenet)
        model = OriginalStyleModel(n_medications=4)
        trainer = OriginalGAMENetTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            loss_fn=OriginalGAMENetLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            metrics=[MeanLogitMetric()],
            target_metric="MeanLogit",
            device="cpu",
            epochs=1,
        )

        results = trainer.fit()

        assert results.final_train_loss is not None
        assert "MeanLogit" in results.best_val_metrics
