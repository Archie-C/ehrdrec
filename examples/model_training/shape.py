import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import SHAPEDataset, collate_shape_examples
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.shape import SHAPE
from ehrdrec.models.utils import create_ddi_adjacency_matrix, create_ehr_adjacency_matrix
from ehrdrec.processing import SetSequenceProcessor
from ehrdrec.training import CheckpointLogger, CompositeLogger, TqdmLogger, Trainer
from ehrdrec.training.losses import OriginalGAMENetLoss


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
EMB_DIM = 128
HIDDEN_DIM = 128
DDI_WEIGHT = 0.05
SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        medication_is_multihot=False,
        min_visits=2,
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

    ehr_adj = create_ehr_adjacency_matrix(
        train_frame,
        medication_col="atc_ids",
        n_medications=n_medications,
    )
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=n_medications,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}, DDI adj: {ddi_adj.shape}")

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

    model = SHAPE(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ehr_adjacency_matrix=ehr_adj.float().cpu().numpy(),
        ddi_adjacency_matrix=ddi_adj.float().cpu().numpy(),
        ddi_mask_H=ddi_adj.float().cpu().numpy(),
        embedding_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        device=device,
    ).to(device)

    metrics = [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
        ),
    ]

    logger = CompositeLogger(
        [
            TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="SHAPE"),
            CheckpointLogger(
                checkpoint_dir="checkpoints/shape",
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
        metrics=metrics,
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
        metrics=metrics,
        device=device,
    ).run()

    print("Test metrics:", eval_results.test_metrics)
