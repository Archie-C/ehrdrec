import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import (
    ConsoleLogger,
    CheckpointLogger,
    CompositeLogger,
    StagedJEPATrainer,
    StageConfig,
    FreezeConfig,
    pretrain_target_space,
)
from ehrdrec.training.losses import BCELoss
from ehrdrec.models import DrugJEPA


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()


ATC_LEVEL   = 5
BATCH_SIZE  = 32

JEPA_EPOCHS     = 50
HEAD_EPOCHS     = 20
FINETUNE_EPOCHS = 5

LR_JEPA     = 1e-3
LR_HEAD     = 1e-3
LR_FINETUNE = 1e-4


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")

    processor = MultiHotProcessor()
    processed_data = processor.process(
        data,
        minimum_admissions=2,
        atc_level=ATC_LEVEL,
        force_reload=True,
    )

    medications_vocab = processor.medications_vocab

    print(processed_data.train_frame.columns)

    train_dataset = MultiHotDataset(
        processed_data.train_frame.collect(),
        target_col="medication_multihot",
        feature_cols=["diagnosis_multihot", "procedure_multihot"],
    )

    val_dataset = MultiHotDataset(
        processed_data.val_frame.collect(),
        target_col="medication_multihot",
        feature_cols=["diagnosis_multihot", "procedure_multihot"],
    )

    test_dataset = MultiHotDataset(
        processed_data.test_frame.collect(),
        target_col="medication_multihot",
        feature_cols=["diagnosis_multihot", "procedure_multihot"],
    )

    x, y = train_dataset[0]

    input_size  = x.shape[0]
    output_size = y.shape[0]

    print(f"Input size: {input_size}, Output size: {output_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = DrugJEPA(
        context_dim=input_size,
        num_meds=output_size,
        hidden_dim=512,
        embedding_dim=128,
        vicreg_sim_weight=25.0,
        vicreg_var_weight=25.0,
        vicreg_cov_weight=0.04,
    )

    pretrain_target_space(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=20,
        lr=1e-3,
        pos_weight=5.0,
    )

    loss_fn = BCELoss()

    # Stage 1: JEPA pretraining.
    # Train context_encoder, jepa_predictor, and drug_embeddings jointly.
    # prediction_head is excluded — it plays no role in pretraining.
    jepa_optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters())
        + list(model.jepa_predictor.parameters()),
        lr=LR_JEPA,
        weight_decay=1e-4,
    )

    # Stage 2: supervised head training.
    # context_encoder is frozen by the FreezeConfig below.
    # Only the prediction_head is trained — it learns to map the
    # pretrained context representations to medication logits.
    head_optimizer = torch.optim.Adam(
        model.prediction_head.parameters(),
        lr=LR_HEAD,
    )

    # Stage 3: fine-tuning.
    # context_encoder unfrozen; prediction_head continues training.
    # jepa_predictor and drug_embeddings excluded — their job is done.
    finetune_optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters())
        + list(model.prediction_head.parameters()),
        lr=LR_FINETUNE,
    )

    metrics = [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
            n_medications=output_size,
            atc_level=ATC_LEVEL,
        ),
    ]

    loggers = [
        ConsoleLogger(),
        CheckpointLogger(
            checkpoint_dir="drug_jepa_checkpoints",
            keep_last=True,
        ),
    ]

    logger = CompositeLogger(loggers)

    trainer = StagedJEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        target_metric="Jaccard",
        higher_is_better=True,
        device=device,
        logger=logger,
        stage_config=StageConfig(
            jepa_epochs=JEPA_EPOCHS,
            head_epochs=HEAD_EPOCHS,
            finetune_epochs=FINETUNE_EPOCHS,
            # Stage 2: freeze context_encoder so only the head trains.
            # jepa_predictor and drug_embeddings are also frozen implicitly
            # since they're not in head_optimizer's parameter groups.
            head_freeze=FreezeConfig(
                freeze=["context_encoder"],
                unfreeze_all_before_stage=True,
            ),
            # Stage 3: unfreeze everything, optimizer controls what updates.
            finetune_freeze=FreezeConfig(
                unfreeze_all_before_stage=True,
            ),
        ),
        jepa_optimizer=jepa_optimizer,
        head_optimizer=head_optimizer,
        finetune_optimizer=finetune_optimizer,
    )

    results = trainer.fit()

    print("Training results:")
    print("Best epoch:", results.best_epoch)
    print("Best training metrics:", results.best_train_metrics)
    print("Best validation metrics:", results.best_val_metrics)

    model.load_state_dict(results.best_model_state)

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        metrics=metrics,
        device=device,
    )

    eval_results = evaluator.run()

    print("Evaluation results:")
    print(eval_results.test_metrics)