import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import OriginalGAMENetDataset, collate_original_gamenet
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.GAMENet import GAMENet
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import (
    CheckpointLogger,
    CompositeLogger,
    OriginalGAMENetTrainer,
    TqdmLogger,
)
from ehrdrec.training.losses import OriginalGAMENetLoss


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
LOOK_BACK = 3
BATCH_SIZE = 1
EPOCHS = 40
LR = 0.002718469948721719
EMB_DIM = 256
DDI_WEIGHT = 0.00
SEED = 42


def evaluate_original_gamenet(model, test_loader, metrics, device):
    for metric in metrics:
        metric.reset()

    model.eval()
    with torch.no_grad():
        for histories, targets in test_loader:
            targets = targets.to(device)
            logits = torch.cat([model(history) for history in histories], dim=0)

            for metric in metrics:
                metric.update(logits, targets)

    return {metric.name: metric.compute() for metric in metrics}


if __name__ == "__main__":
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

    dataset_kwargs = dict(
        target_col="medication_multihot",
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )

    train_dataset = OriginalGAMENetDataset(train_frame, **dataset_kwargs)
    val_dataset = OriginalGAMENetDataset(val_frame, **dataset_kwargs)
    test_dataset = OriginalGAMENetDataset(test_frame, **dataset_kwargs)

    _, sample_y = train_dataset[0]
    output_size = sample_y.shape[0]
    print(f"Output size: {output_size}")

    ehr_adj = create_ehr_adjacency_matrix(train_frame)
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}, DDI adj: {ddi_adj.shape}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_original_gamenet,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_original_gamenet,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_original_gamenet,
    )

    model = GAMENet(
        vocab_size=[
            len(processor.diagnoses_vocab.id_to_token),
            len(processor.procedures_vocab.id_to_token),
            output_size,
        ],
        ehr_adj=ehr_adj.cpu().numpy(),
        ddi_adj=ddi_adj.float().cpu().numpy(),
        emb_dim=EMB_DIM,
        device=device,
        ddi_in_memory=True,
    )

    metrics = [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=processor.medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
    ]

    logger = CompositeLogger(
        [
            TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="Original GAMENet"),
            CheckpointLogger(
                checkpoint_dir="checkpoints/original_gamenet",
                keep_last=True,
            ),
        ]
    )

    trainer = OriginalGAMENetTrainer(
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

    test_metrics = evaluate_original_gamenet(
        model=model,
        test_loader=test_loader,
        metrics=metrics,
        device=device,
    )

    print("Test metrics:", test_metrics)
