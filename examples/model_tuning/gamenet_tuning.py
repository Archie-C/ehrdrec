import logging

import optuna
import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import MultiHotDatasetWithPatientLookBack, collate_patient_visit_histories
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC4Loader, MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import GameNetFast
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, Tuner, TqdmLogger, TunerTqdmCallback
from ehrdrec.training.losses import BCELoss

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()
optuna.logging.set_verbosity(optuna.logging.WARNING)

ATC_LEVEL = 5
LOOK_BACK = 3
N_TRIALS = 20
TUNE_EPOCHS = 15
FINAL_EPOCHS = 40

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)

    diagnoses_vocab = processor.diagnoses_vocab
    procedures_vocab = processor.procedures_vocab
    medications_vocab = processor.medications_vocab

    train_frame = processed_data.train_frame.collect()
    val_frame = processed_data.val_frame.collect()
    test_frame = processed_data.test_frame.collect()

    dataset_kwargs = dict(
        target_col="medication_multihot",
        n_diagnoses=len(diagnoses_vocab.id_to_token),
        n_procedures=len(procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )

    train_dataset = MultiHotDatasetWithPatientLookBack(train_frame, **dataset_kwargs)
    val_dataset = MultiHotDatasetWithPatientLookBack(val_frame, **dataset_kwargs)
    test_dataset = MultiHotDatasetWithPatientLookBack(test_frame, **dataset_kwargs)

    _, sample_y = train_dataset[0]
    output_size = sample_y.shape[0]
    print(f"Output size: {output_size}")

    # Adjacency matrices don't depend on hyperparameters — build once and reuse across trials.
    ehr_adj_matrix = create_ehr_adjacency_matrix(train_frame)
    ddi_adj_matrix = create_ddi_adjacency_matrix(
        medications_vocab=medications_vocab,
        ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adjacency matrix shape: {ehr_adj_matrix.shape}")
    print(f"DDI adjacency matrix shape: {ddi_adj_matrix.shape}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_patient_visit_histories)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_patient_visit_histories)

    ddi_kwargs = dict(
        medications_vocab=medications_vocab,
        ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )

    # Cheap metrics only — DDI is too expensive to run on every tuning val pass
    def make_tuning_metrics():
        return [Jaccard(), F1(), PRAUC()]

    def make_full_metrics():
        return [
            Jaccard(),
            F1(),
            PRAUC(),
            BinaryDDI(**ddi_kwargs),
            HighSeverityBinaryDDI(**ddi_kwargs),
        ]

    def trial_fn(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader) -> Trainer:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        query_dim = trial.suggest_categorical("query_dim", [64, 128, 256])

        model = GameNetFast(
            n_diagnoses=len(diagnoses_vocab.id_to_token),
            n_procedures=len(procedures_vocab.id_to_token),
            n_medications=output_size,
            medication_adjacency_matrix=ehr_adj_matrix,
            ddi_adjacency_matrix=ddi_adj_matrix,
            diagnoses_embedding_dim=embedding_dim,
            procedures_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            query_dim=query_dim,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=BCELoss(),
            optimizer=optimizer,
            metrics=make_tuning_metrics(),
            target_metric="Jaccard",
            higher_is_better=True,
            device=device,
            epochs=TUNE_EPOCHS,
            logger=TqdmLogger(epochs=TUNE_EPOCHS, metrics=["Jaccard"], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    tuner = Tuner(
        trial_fn,
        n_trials=N_TRIALS,
        direction="maximize",
        callbacks=[TunerTqdmCallback(n_trials=N_TRIALS, direction="maximize")],
    )
    study, best_results = tuner.tune(train_loader, val_loader)

    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation Jaccard: {study.best_value:.4f}")

    # Retrain with best params for full epochs, then evaluate on test
    best_params = study.best_params
    model = GameNetFast(
        n_diagnoses=len(diagnoses_vocab.id_to_token),
        n_procedures=len(procedures_vocab.id_to_token),
        n_medications=output_size,
        medication_adjacency_matrix=ehr_adj_matrix,
        ddi_adjacency_matrix=ddi_adj_matrix,
        diagnoses_embedding_dim=best_params["embedding_dim"],
        procedures_embedding_dim=best_params["embedding_dim"],
        hidden_dim=best_params["hidden_dim"],
        query_dim=best_params["query_dim"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=BCELoss(),
        optimizer=optimizer,
        metrics=make_full_metrics(),
        target_metric="Jaccard",
        higher_is_better=True,
        device=device,
        epochs=FINAL_EPOCHS,
        logger=TqdmLogger(epochs=FINAL_EPOCHS, metrics=["Jaccard"], desc="Final training"),
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    evaluator = Evaluator(
        model=model,
        test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_patient_visit_histories),
        metrics=make_full_metrics(),
        device=device,
    )
    eval_results = evaluator.run()
    print("\nTest evaluation results:")
    print(eval_results.test_metrics)
