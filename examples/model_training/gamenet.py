import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import MultiHotDatasetWithPatientLookBack, collate_patient_visit_histories
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import GameNetFast
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, ConsoleLogger, CheckpointLogger, CompositeLogger
from ehrdrec.training.losses import BCELoss


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL      = 5
LOOK_BACK      = 3
BATCH_SIZE     = 32
EPOCHS         = 40
LR             = 5e-4


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")

    processor = MultiHotProcessor()
    processed = processor.process(
        data,
        minimum_admissions=2,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )

    train_frame = processed.train_frame.collect()
    val_frame   = processed.val_frame.collect()
    test_frame  = processed.test_frame.collect()

    dataset_kwargs = dict(
        target_col="medication_multihot",
        n_diagnoses=len(processor.diagnoses_vocab.id_to_token),
        n_procedures=len(processor.procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )

    train_dataset = MultiHotDatasetWithPatientLookBack(train_frame, **dataset_kwargs)
    val_dataset   = MultiHotDatasetWithPatientLookBack(val_frame,   **dataset_kwargs)
    test_dataset  = MultiHotDatasetWithPatientLookBack(test_frame,  **dataset_kwargs)

    _, sample_y = train_dataset[0]
    output_size = sample_y.shape[0]
    print(f"Output size: {output_size}")

    ehr_adj = create_ehr_adjacency_matrix(train_frame)
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}, DDI adj: {ddi_adj.shape}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_patient_visit_histories)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_patient_visit_histories)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_patient_visit_histories)

    model = GameNetFast(
        n_diagnoses=len(processor.diagnoses_vocab.id_to_token),
        n_procedures=len(processor.procedures_vocab.id_to_token),
        n_medications=output_size,
        medication_adjacency_matrix=ehr_adj,
        ddi_adjacency_matrix=ddi_adj,
        diagnoses_embedding_dim=128,
        procedures_embedding_dim=128,
        hidden_dim=128,
        query_dim=128,
    )

    metrics = [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
    ]

    logger = CompositeLogger([
        ConsoleLogger(),
        CheckpointLogger(checkpoint_dir="gamenet_checkpoints", keep_last=True),
    ])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=BCELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        metrics=metrics,
        target_metric="Jaccard",
        higher_is_better=True,
        device=device,
        epochs=EPOCHS,
        logger=logger,
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
