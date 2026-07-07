import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.datasets import MultiHotDatasetWithPatientLookBack, collate_patient_visit_histories
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import BinaryDDI, F1, Jaccard, PRAUC
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.sspnet import SSPNet
from ehrdrec.models.utils import create_ddi_adjacency_matrix, create_ehr_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import CheckpointLogger, CompositeLogger, TqdmLogger, Trainer
from ehrdrec.training.losses import OriginalGAMENetLoss


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
LOOK_BACK = 3
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
EMB_DIM = 128
N_HEADS = 4
DROPOUT = 0.5
DDI_WEIGHT = 0.05
SEED = 42


class SSPNetTrainingAdapter(nn.Module):
    """Adapts padded multihot history batches to original SSPNet inputs."""

    def __init__(self, model: SSPNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: dict[str, torch.Tensor]):
        predictions = []
        ddi_losses = []
        batch_size = features["diagnoses"].size(0)

        for batch_idx in range(batch_size):
            length = int(features["lengths"][batch_idx].item())
            sample_features = {
                "diagnoses": self._multihot_visits_to_padded_ids(features["diagnoses"][batch_idx, :length]),
                "procedures": self._multihot_visits_to_padded_ids(features["procedures"][batch_idx, :length]),
                "medication_history": features["medication_history"][batch_idx : batch_idx + 1, :length],
            }
            output = self.model(sample_features)
            predictions.append(output["predictions"])

            losses = output.get("losses")
            if losses is not None and "ddi_loss" in losses:
                ddi_losses.append(losses["ddi_loss"])

        result = {"predictions": torch.cat(predictions, dim=0)}
        if ddi_losses:
            result["losses"] = {"ddi_loss": torch.stack(ddi_losses).mean()}
        return result

    @staticmethod
    def _multihot_visits_to_padded_ids(visits: torch.Tensor) -> torch.Tensor:
        visit_ids = [visit.nonzero(as_tuple=False).flatten().long() for visit in visits]
        max_codes = max((ids.numel() for ids in visit_ids), default=1)
        max_codes = max(max_codes, 1)
        padded = visits.new_zeros((1, len(visit_ids), max_codes), dtype=torch.long)
        for visit_idx, ids in enumerate(visit_ids):
            if ids.numel() > 0:
                padded[0, visit_idx, : ids.numel()] = ids
        return padded


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_metrics(medications_vocab, n_medications: int, device: torch.device) -> list:
    return [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
            device=device,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
            device=device,
        ),
    ]


def build_sspnet_model(
    *,
    n_diagnoses: int,
    n_procedures: int,
    n_medications: int,
    ehr_adj: torch.Tensor,
    ddi_adj: torch.Tensor,
    device: torch.device,
) -> SSPNetTrainingAdapter:
    sspnet = SSPNet(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ehr_adjacency_matrix=ehr_adj,
        ddi_adjacency_matrix=ddi_adj,
        embedding_dim=EMB_DIM,
        number_of_heads=N_HEADS,
        dropout=DROPOUT,
        device=device,
    )
    return SSPNetTrainingAdapter(sspnet).to(device)


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = MIMIC3Loader()
    data = loader.load(MIMIC3_PATH)

    processor = MultiHotProcessor()
    processed = processor.process(
        data,
        minimum_admissions=2,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )

    train_frame = processed.train_frame.collect()
    val_frame = processed.val_frame.collect()
    test_frame = processed.test_frame.collect()

    n_diagnoses = processor.diagnoses_vocab.vocab_size
    n_procedures = processor.procedures_vocab.vocab_size
    n_medications = processor.medications_vocab.vocab_size
    print(
        "Vocab sizes:",
        f"diagnoses={n_diagnoses}",
        f"procedures={n_procedures}",
        f"medications={n_medications}",
    )

    dataset_kwargs = dict(
        target_col="medication_multihot",
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    train_dataset = MultiHotDatasetWithPatientLookBack(train_frame, **dataset_kwargs)
    val_dataset = MultiHotDatasetWithPatientLookBack(val_frame, **dataset_kwargs)
    test_dataset = MultiHotDatasetWithPatientLookBack(test_frame, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_patient_visit_histories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_patient_visit_histories,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_patient_visit_histories,
    )

    ehr_adj = create_ehr_adjacency_matrix(
        train_frame,
        medication_col="medication_multihot",
        n_medications=n_medications,
    )
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=n_medications,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}")
    print(f"DDI adj: {ddi_adj.shape}")

    model = build_sspnet_model(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ehr_adj=ehr_adj,
        ddi_adj=ddi_adj,
        device=device,
    )

    logger = CompositeLogger(
        [
            TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="SSPNet"),
            CheckpointLogger(
                checkpoint_dir="checkpoints/sspnet",
                keep_last=True,
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=OriginalGAMENetLoss(ddi_weight=DDI_WEIGHT),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        metrics=make_metrics(processor.medications_vocab, n_medications, device),
        target_metric="Jaccard",
        higher_is_better=True,
        device=device,
        epochs=EPOCHS,
        logger=logger,
        seed=SEED,
    )

    results = trainer.fit()

    print("Best epoch:", results.best_epoch)
    print("Best train metrics:", results.best_train_metrics)
    print("Best val metrics:", results.best_val_metrics)

    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model,
        test_loader=test_loader,
        metrics=make_metrics(processor.medications_vocab, n_medications, device),
        device=device,
    ).run()

    print("Test metrics:", eval_results.test_metrics)
