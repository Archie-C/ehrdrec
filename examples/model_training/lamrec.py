import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.datasets import SHAPEDataset, collate_shape_examples
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import BinaryDDI, F1, Jaccard, PRAUC
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.LAMRec import LAMRec
from ehrdrec.models.utils import create_ddi_adjacency_matrix
from ehrdrec.processing import SetSequenceProcessor
from ehrdrec.training import CheckpointLogger, CompositeLogger, TqdmLogger, Trainer


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
EMB_DIM = 128
N_HEADS = 4
NUM_LAYERS = 2
FEEDFORWARD_DIM = 128
TEMPERATURE = 0.07
DDI_WEIGHT = 0.5
MULTIVIEW_WEIGHT = 0.5
SEED = 42
LOOK_BACK = None
MIN_VISITS = 2
DDI_THRESHOLD = 0.06


class LAMRecTrainingAdapter(nn.Module):
    """Adapts SHAPEDataset batches to the original LAMRec input contract."""

    def __init__(self, model: LAMRec) -> None:
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model(
            {
                "diagnoses": features["diseases"],
                "procedures": features["procedures"],
            }
        )


class LAMRecLoss(nn.Module):
    def __init__(self, ddi_weight: float = 0.5, multiview_weight: float = 0.5) -> None:
        super().__init__()
        self.ddi_weight = ddi_weight
        self.multiview_weight = multiview_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, model_output=None, features=None, losses=None, **kwargs):
        loss = self.bce_loss(predictions, targets)
        if losses is None:
            return loss

        if "ddi_loss" in losses:
            loss = loss + self.ddi_weight * losses["ddi_loss"].to(predictions.device)
        if "multiview_loss" in losses:
            loss = loss + self.multiview_weight * losses["multiview_loss"].to(predictions.device)

        return loss


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


def build_lamrec_model(
    *,
    n_diagnoses: int,
    n_procedures: int,
    n_medications: int,
    ddi_adj: torch.Tensor,
) -> LAMRecTrainingAdapter:
    lamrec = LAMRec(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ddi_adjacency_matrix=ddi_adj,
        embedding_dim=EMB_DIM,
        alpha=DDI_WEIGHT,
        beta=MULTIVIEW_WEIGHT,
        number_of_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        temperature=TEMPERATURE,
        ddi_threshold=DDI_THRESHOLD,
    )
    return LAMRecTrainingAdapter(lamrec)


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = MIMIC3Loader()
    data = loader.load(MIMIC3_PATH)

    processor = SetSequenceProcessor()
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

    dataset_kwargs = dict(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        patient_id_col="patient_id",
        time_col="admission_time",
        diagnosis_col="diagnosis_ids",
        procedure_col="procedure_ids",
        medication_col="atc_ids",
        min_visits=MIN_VISITS,
        sample_all_visits=True,
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )

    train_dataset = SHAPEDataset(train_frame, **dataset_kwargs)
    val_dataset = SHAPEDataset(val_frame, **dataset_kwargs)
    test_dataset = SHAPEDataset(test_frame, **dataset_kwargs)

    print(
        "Vocab sizes:",
        f"diagnoses={n_diagnoses}",
        f"procedures={n_procedures}",
        f"medications={n_medications}",
    )

    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=n_medications,
        atc_level=ATC_LEVEL,
    )
    print(f"DDI adj: {ddi_adj.shape}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_shape_examples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_shape_examples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_shape_examples,
    )

    model = build_lamrec_model(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ddi_adj=ddi_adj,
    )

    logger = CompositeLogger(
        [
            TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="LAMRec"),
            CheckpointLogger(
                checkpoint_dir="checkpoints/lamrec",
                keep_last=True,
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=LAMRecLoss(ddi_weight=DDI_WEIGHT, multiview_weight=MULTIVIEW_WEIGHT),
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
