"""Overnight benchmark: tune all models on MIMIC-III and save full run artefacts.

Each model is tuned independently with Optuna, retrained with the best params
for full epochs, evaluated on the held-out test set, and saved to its own
subdirectory under OUTPUT_DIR via save_run().

Directory layout:
    outputs/mimic3_benchmark/
        mlp/
        foursdrug/
        gamenet/
        fastrx/
        micron/
        comparison.json   <- side-by-side test metrics for all models
"""

import json
import logging
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import (
    MultiHotDataset,
    MultiHotDatasetWithPatientLookBack,
    collate_patient_visit_histories,
)
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import MLP, FourSDrug, GameNetFast, FastRx, Micron, ExperimentConfig
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, Tuner, TqdmLogger, TunerTqdmCallback
from ehrdrec.training.losses import BCELoss, MicronLoss
from ehrdrec.utils import save_run

# ── global settings ──────────────────────────────────────────────────────────

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_PATH     = "/home/cararc/data/mimic-iii-1.4"
ATC_LEVEL        = 5
MINIMUM_ADMISSIONS = 2
LOOK_BACK        = 3
BATCH_SIZE       = 32
BATCH_SIZE_FASTRX = 256          # FastRx uses larger batches
DDI_PATH         = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"
SEED             = 42
TARGET_METRIC    = "Jaccard"
OUTPUT_DIR       = Path("outputs/mimic3_benchmark")

# Per-model trial / epoch budgets
TUNE_EPOCHS  = {"mlp": 20, "foursdrug": 20, "gamenet": 15, "fastrx": 15, "micron": 15}
FINAL_EPOCHS = {"mlp": 40, "foursdrug": 40, "gamenet": 40, "fastrx": 100, "micron": 40}
N_TRIALS     = {"mlp": 30, "foursdrug": 30, "gamenet": 20, "fastrx": 20, "micron": 20}

# ── shared data loading ───────────────────────────────────────────────────────

def load_data():
    loader = MIMIC3Loader()
    data = loader.load(DATASET_PATH)
    processor = MultiHotProcessor()
    processed = processor.process(
        data,
        minimum_admissions=MINIMUM_ADMISSIONS,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )
    return processed, processor


def make_ddi_kwargs(medications_vocab, output_size):
    return dict(
        medications_vocab=medications_vocab,
        ddinter_path=DDI_PATH,
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )


def make_tuning_metrics():
    return [Jaccard(), F1(), PRAUC()]


def make_full_metrics(ddi_kwargs):
    return [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(**ddi_kwargs),
        HighSeverityBinaryDDI(**ddi_kwargs),
    ]


# ── per-model tuning functions ────────────────────────────────────────────────

def tune_mlp(processed, processor, device):
    name = "mlp"
    print(f"\n{'='*60}\nTuning {name.upper()}\n{'='*60}")

    n_diag = len(processor.diagnoses_vocab.id_to_token)
    n_proc = len(processor.procedures_vocab.id_to_token)
    medications_vocab = processor.medications_vocab

    ds_kwargs = dict(target_col="medication_multihot", n_diagnoses=n_diag, n_procedures=n_proc)
    train_ds = MultiHotDataset(processed.train_frame.collect(), **ds_kwargs)
    val_ds   = MultiHotDataset(processed.val_frame.collect(),   **ds_kwargs)
    test_ds  = MultiHotDataset(processed.test_frame.collect(),  **ds_kwargs)

    x, y = train_ds[0]
    input_size, output_size = x.shape[0], y.shape[0]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    ddi_kwargs   = make_ddi_kwargs(medications_vocab, output_size)

    def trial_fn(trial, train_loader, val_loader):
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout     = trial.suggest_float("dropout", 0.1, 0.6)
        n_layers    = trial.suggest_int("n_layers", 2, 4)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        model = MLP(input_size, [hidden_size] * n_layers, output_size, dropout)
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            metrics=make_tuning_metrics(), target_metric=TARGET_METRIC,
            higher_is_better=True, device=device, epochs=TUNE_EPOCHS[name],
            logger=TqdmLogger(epochs=TUNE_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    study, _ = Tuner(trial_fn, n_trials=N_TRIALS[name], direction="maximize", seed=SEED,
                     callbacks=[TunerTqdmCallback(n_trials=N_TRIALS[name], direction="maximize")],
                     ).tune(train_loader, val_loader)

    bp = study.best_params
    hidden_sizes = [bp["hidden_size"]] * bp["n_layers"]
    model = MLP(input_size, hidden_sizes, output_size, bp["dropout"])
    config = ExperimentConfig(
        dataset_path=DATASET_PATH, dataset_name="mimic-iii",
        atc_level=ATC_LEVEL, minimum_admissions=MINIMUM_ADMISSIONS,
        input_size=input_size, output_size=output_size,
        batch_size=BATCH_SIZE, epochs=FINAL_EPOCHS[name],
        lr=bp["lr"], seed=SEED,
        n_tuning_trials=N_TRIALS[name], tuning_epochs=TUNE_EPOCHS[name], tuning_metric=TARGET_METRIC,
        model_kwargs={"dropout": bp["dropout"], "n_layers": bp["n_layers"],
                      "hidden_size": bp["hidden_size"], "hidden_sizes": hidden_sizes},
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=bp["lr"]),
        metrics=make_full_metrics(ddi_kwargs), target_metric=TARGET_METRIC,
        higher_is_better=True, device=device, epochs=FINAL_EPOCHS[name],
        logger=TqdmLogger(epochs=FINAL_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"[{name}] Final"),
        seed=SEED,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model, test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        metrics=make_full_metrics(ddi_kwargs), device=device, save_predictions=True,
    ).run()

    save_run(OUTPUT_DIR / name, config=config, training_results=results, eval_results=eval_results,
             study=study, vocabs={"medications": medications_vocab,
                                  "diagnoses": processor.diagnoses_vocab,
                                  "procedures": processor.procedures_vocab})
    return eval_results.test_metrics


def tune_foursdrug(processed, processor, device):
    name = "foursdrug"
    print(f"\n{'='*60}\nTuning {name.upper()}\n{'='*60}")

    n_diag = len(processor.diagnoses_vocab.id_to_token)
    n_proc = len(processor.procedures_vocab.id_to_token)
    medications_vocab = processor.medications_vocab

    ds_kwargs = dict(target_col="medication_multihot", n_diagnoses=n_diag, n_procedures=n_proc)
    train_ds = MultiHotDataset(processed.train_frame.collect(), **ds_kwargs)
    val_ds   = MultiHotDataset(processed.val_frame.collect(),   **ds_kwargs)
    test_ds  = MultiHotDataset(processed.test_frame.collect(),  **ds_kwargs)

    x, y = train_ds[0]
    input_size, output_size = x.shape[0], y.shape[0]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    ddi_kwargs   = make_ddi_kwargs(medications_vocab, output_size)

    def trial_fn(trial, train_loader, val_loader):
        lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128, 256])
        model   = FourSDrug(num_symptoms=input_size, num_drugs=output_size, emb_dim=emb_dim)
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            metrics=make_tuning_metrics(), target_metric=TARGET_METRIC,
            higher_is_better=True, device=device, epochs=TUNE_EPOCHS[name],
            logger=TqdmLogger(epochs=TUNE_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    study, _ = Tuner(trial_fn, n_trials=N_TRIALS[name], direction="maximize", seed=SEED,
                     callbacks=[TunerTqdmCallback(n_trials=N_TRIALS[name], direction="maximize")],
                     ).tune(train_loader, val_loader)

    bp = study.best_params
    model = FourSDrug(num_symptoms=input_size, num_drugs=output_size, emb_dim=bp["emb_dim"])
    config = ExperimentConfig(
        dataset_path=DATASET_PATH, dataset_name="mimic-iii",
        atc_level=ATC_LEVEL, minimum_admissions=MINIMUM_ADMISSIONS,
        input_size=input_size, output_size=output_size,
        batch_size=BATCH_SIZE, epochs=FINAL_EPOCHS[name],
        lr=bp["lr"], seed=SEED,
        n_tuning_trials=N_TRIALS[name], tuning_epochs=TUNE_EPOCHS[name], tuning_metric=TARGET_METRIC,
        model_kwargs={"emb_dim": bp["emb_dim"]},
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=bp["lr"]),
        metrics=make_full_metrics(ddi_kwargs), target_metric=TARGET_METRIC,
        higher_is_better=True, device=device, epochs=FINAL_EPOCHS[name],
        logger=TqdmLogger(epochs=FINAL_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"[{name}] Final"),
        seed=SEED,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model, test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
        metrics=make_full_metrics(ddi_kwargs), device=device, save_predictions=True,
    ).run()

    save_run(OUTPUT_DIR / name, config=config, training_results=results, eval_results=eval_results,
             study=study, vocabs={"medications": medications_vocab,
                                  "diagnoses": processor.diagnoses_vocab,
                                  "procedures": processor.procedures_vocab})
    return eval_results.test_metrics


def tune_gamenet(processed, processor, device, ehr_adj, ddi_adj):
    name = "gamenet"
    print(f"\n{'='*60}\nTuning {name.upper()}\n{'='*60}")

    n_diag = len(processor.diagnoses_vocab.id_to_token)
    n_proc = len(processor.procedures_vocab.id_to_token)
    medications_vocab = processor.medications_vocab

    ds_kwargs = dict(
        target_col="medication_multihot", n_diagnoses=n_diag, n_procedures=n_proc,
        patient_id_col="patient_id", time_col="admission_time",
        look_back=LOOK_BACK, dtype=torch.float32,
    )
    train_ds = MultiHotDatasetWithPatientLookBack(processed.train_frame.collect(), **ds_kwargs)
    val_ds   = MultiHotDatasetWithPatientLookBack(processed.val_frame.collect(),   **ds_kwargs)
    test_ds  = MultiHotDatasetWithPatientLookBack(processed.test_frame.collect(),  **ds_kwargs)

    _, sample_y = train_ds[0]
    output_size = sample_y.shape[0]

    collate = collate_patient_visit_histories
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    ddi_kwargs   = make_ddi_kwargs(medications_vocab, output_size)

    def trial_fn(trial, train_loader, val_loader):
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        hidden_dim    = trial.suggest_categorical("hidden_dim",    [64, 128, 256])
        query_dim     = trial.suggest_categorical("query_dim",     [64, 128, 256])
        model = GameNetFast(
            n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
            medication_adjacency_matrix=ehr_adj, ddi_adjacency_matrix=ddi_adj,
            diagnoses_embedding_dim=embedding_dim, procedures_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, query_dim=query_dim,
        )
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            metrics=make_tuning_metrics(), target_metric=TARGET_METRIC,
            higher_is_better=True, device=device, epochs=TUNE_EPOCHS[name],
            logger=TqdmLogger(epochs=TUNE_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    study, _ = Tuner(trial_fn, n_trials=N_TRIALS[name], direction="maximize", seed=SEED,
                     callbacks=[TunerTqdmCallback(n_trials=N_TRIALS[name], direction="maximize")],
                     ).tune(train_loader, val_loader)

    bp = study.best_params
    model = GameNetFast(
        n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
        medication_adjacency_matrix=ehr_adj, ddi_adjacency_matrix=ddi_adj,
        diagnoses_embedding_dim=bp["embedding_dim"], procedures_embedding_dim=bp["embedding_dim"],
        hidden_dim=bp["hidden_dim"], query_dim=bp["query_dim"],
    )
    config = ExperimentConfig(
        dataset_path=DATASET_PATH, dataset_name="mimic-iii",
        atc_level=ATC_LEVEL, minimum_admissions=MINIMUM_ADMISSIONS,
        output_size=output_size, batch_size=BATCH_SIZE, epochs=FINAL_EPOCHS[name],
        lr=bp["lr"], seed=SEED,
        n_tuning_trials=N_TRIALS[name], tuning_epochs=TUNE_EPOCHS[name], tuning_metric=TARGET_METRIC,
        model_kwargs={"embedding_dim": bp["embedding_dim"], "hidden_dim": bp["hidden_dim"],
                      "query_dim": bp["query_dim"], "look_back": LOOK_BACK},
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=bp["lr"]),
        metrics=make_full_metrics(ddi_kwargs), target_metric=TARGET_METRIC,
        higher_is_better=True, device=device, epochs=FINAL_EPOCHS[name],
        logger=TqdmLogger(epochs=FINAL_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"[{name}] Final"),
        seed=SEED,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model,
        test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate),
        metrics=make_full_metrics(ddi_kwargs), device=device, save_predictions=True,
    ).run()

    save_run(OUTPUT_DIR / name, config=config, training_results=results, eval_results=eval_results,
             study=study, vocabs={"medications": medications_vocab,
                                  "diagnoses": processor.diagnoses_vocab,
                                  "procedures": processor.procedures_vocab})
    return eval_results.test_metrics


def tune_fastrx(processed, processor, device, ehr_adj, ddi_adj):
    name = "fastrx"
    print(f"\n{'='*60}\nTuning {name.upper()}\n{'='*60}")

    n_diag = len(processor.diagnoses_vocab.id_to_token)
    n_proc = len(processor.procedures_vocab.id_to_token)
    medications_vocab = processor.medications_vocab

    ds_kwargs = dict(
        target_col="medication_multihot", n_diagnoses=n_diag, n_procedures=n_proc,
        patient_id_col="patient_id", time_col="admission_time",
        look_back=LOOK_BACK, dtype=torch.float32,
    )
    train_ds = MultiHotDatasetWithPatientLookBack(processed.train_frame.collect(), **ds_kwargs)
    val_ds   = MultiHotDatasetWithPatientLookBack(processed.val_frame.collect(),   **ds_kwargs)
    test_ds  = MultiHotDatasetWithPatientLookBack(processed.test_frame.collect(),  **ds_kwargs)

    _, sample_y = train_ds[0]
    output_size = sample_y.shape[0]

    collate = collate_patient_visit_histories
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_FASTRX, shuffle=False, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_FASTRX, shuffle=False, collate_fn=collate)
    ddi_kwargs   = make_ddi_kwargs(medications_vocab, output_size)

    def trial_fn(trial, train_loader, val_loader):
        lr                      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout                 = trial.suggest_float("dropout", 0.1, 0.6)
        embedding_dim           = trial.suggest_categorical("embedding_dim",           [64, 128, 256, 512])
        embedding_dim_fastformer = trial.suggest_categorical("embedding_dim_fastformer", [64, 128, 256])
        model = FastRx(
            n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
            medication_adjacency_matrix=ehr_adj, ddi_adjacency_matrix=ddi_adj,
            embedding_dim=embedding_dim, embedding_dim_fastformer=embedding_dim_fastformer,
            dropout=dropout,
        )
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            metrics=make_tuning_metrics(), target_metric=TARGET_METRIC,
            higher_is_better=True, device=device, epochs=TUNE_EPOCHS[name],
            logger=TqdmLogger(epochs=TUNE_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    study, _ = Tuner(trial_fn, n_trials=N_TRIALS[name], direction="maximize", seed=SEED,
                     callbacks=[TunerTqdmCallback(n_trials=N_TRIALS[name], direction="maximize")],
                     ).tune(train_loader, val_loader)

    bp = study.best_params
    model = FastRx(
        n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
        medication_adjacency_matrix=ehr_adj, ddi_adjacency_matrix=ddi_adj,
        embedding_dim=bp["embedding_dim"], embedding_dim_fastformer=bp["embedding_dim_fastformer"],
        dropout=bp["dropout"],
    )
    config = ExperimentConfig(
        dataset_path=DATASET_PATH, dataset_name="mimic-iii",
        atc_level=ATC_LEVEL, minimum_admissions=MINIMUM_ADMISSIONS,
        output_size=output_size, batch_size=BATCH_SIZE_FASTRX, epochs=FINAL_EPOCHS[name],
        lr=bp["lr"], seed=SEED,
        n_tuning_trials=N_TRIALS[name], tuning_epochs=TUNE_EPOCHS[name], tuning_metric=TARGET_METRIC,
        model_kwargs={"embedding_dim": bp["embedding_dim"],
                      "embedding_dim_fastformer": bp["embedding_dim_fastformer"],
                      "dropout": bp["dropout"], "look_back": LOOK_BACK},
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=BCELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=bp["lr"]),
        metrics=make_full_metrics(ddi_kwargs), target_metric=TARGET_METRIC,
        higher_is_better=True, device=device, epochs=FINAL_EPOCHS[name],
        logger=TqdmLogger(epochs=FINAL_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"[{name}] Final"),
        seed=SEED,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model,
        test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE_FASTRX, shuffle=False, collate_fn=collate),
        metrics=make_full_metrics(ddi_kwargs), device=device, save_predictions=True,
    ).run()

    save_run(OUTPUT_DIR / name, config=config, training_results=results, eval_results=eval_results,
             study=study, vocabs={"medications": medications_vocab,
                                  "diagnoses": processor.diagnoses_vocab,
                                  "procedures": processor.procedures_vocab})
    return eval_results.test_metrics


def tune_micron(processed, processor, device, ddi_adj):
    name = "micron"
    print(f"\n{'='*60}\nTuning {name.upper()}\n{'='*60}")

    n_diag = len(processor.diagnoses_vocab.id_to_token)
    n_proc = len(processor.procedures_vocab.id_to_token)
    medications_vocab = processor.medications_vocab

    ds_kwargs = dict(
        target_col="medication_multihot", n_diagnoses=n_diag, n_procedures=n_proc,
        patient_id_col="patient_id", time_col="admission_time",
        look_back=LOOK_BACK, dtype=torch.float32,
    )
    train_ds = MultiHotDatasetWithPatientLookBack(processed.train_frame.collect(), **ds_kwargs)
    val_ds   = MultiHotDatasetWithPatientLookBack(processed.val_frame.collect(),   **ds_kwargs)
    test_ds  = MultiHotDatasetWithPatientLookBack(processed.test_frame.collect(),  **ds_kwargs)

    _, sample_y = train_ds[0]
    output_size = sample_y.shape[0]

    collate = collate_patient_visit_histories
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    ddi_kwargs   = make_ddi_kwargs(medications_vocab, output_size)

    def trial_fn(trial, train_loader, val_loader):
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout       = trial.suggest_float("dropout", 0.1, 0.6)
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        alpha         = trial.suggest_float("alpha", 0.3, 0.9)
        model = Micron(
            n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
            ddi_adjacency_matrix=ddi_adj, embedding_dim=embedding_dim,
            dropout=dropout, return_losses=True,
        )
        return Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=MicronLoss(alpha=alpha), optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            metrics=make_tuning_metrics(), target_metric=TARGET_METRIC,
            higher_is_better=True, device=device, epochs=TUNE_EPOCHS[name],
            logger=TqdmLogger(epochs=TUNE_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"Trial {trial.number}"),
            trial=trial,
        )

    study, _ = Tuner(trial_fn, n_trials=N_TRIALS[name], direction="maximize", seed=SEED,
                     callbacks=[TunerTqdmCallback(n_trials=N_TRIALS[name], direction="maximize")],
                     ).tune(train_loader, val_loader)

    bp = study.best_params
    model = Micron(
        n_diagnoses=n_diag, n_procedures=n_proc, n_medications=output_size,
        ddi_adjacency_matrix=ddi_adj, embedding_dim=bp["embedding_dim"],
        dropout=bp["dropout"], return_losses=True,
    )
    config = ExperimentConfig(
        dataset_path=DATASET_PATH, dataset_name="mimic-iii",
        atc_level=ATC_LEVEL, minimum_admissions=MINIMUM_ADMISSIONS,
        output_size=output_size, batch_size=BATCH_SIZE, epochs=FINAL_EPOCHS[name],
        lr=bp["lr"], seed=SEED,
        n_tuning_trials=N_TRIALS[name], tuning_epochs=TUNE_EPOCHS[name], tuning_metric=TARGET_METRIC,
        model_kwargs={"embedding_dim": bp["embedding_dim"], "dropout": bp["dropout"],
                      "alpha": bp["alpha"], "look_back": LOOK_BACK},
    )
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=MicronLoss(alpha=bp["alpha"]),
        optimizer=torch.optim.Adam(model.parameters(), lr=bp["lr"]),
        metrics=make_full_metrics(ddi_kwargs), target_metric=TARGET_METRIC,
        higher_is_better=True, device=device, epochs=FINAL_EPOCHS[name],
        logger=TqdmLogger(epochs=FINAL_EPOCHS[name], metrics=[TARGET_METRIC], desc=f"[{name}] Final"),
        seed=SEED,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    eval_results = Evaluator(
        model=model,
        test_loader=DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate),
        metrics=make_full_metrics(ddi_kwargs), device=device, save_predictions=True,
    ).run()

    save_run(OUTPUT_DIR / name, config=config, training_results=results, eval_results=eval_results,
             study=study, vocabs={"medications": medications_vocab,
                                  "diagnoses": processor.diagnoses_vocab,
                                  "procedures": processor.procedures_vocab})
    return eval_results.test_metrics


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and process data once — all models share the same processed frames
    print("\nLoading and processing data...")
    processed, processor = load_data()
    medications_vocab = processor.medications_vocab

    # Adjacency matrices needed by graph-based models — build once
    print("\nBuilding adjacency matrices...")
    train_frame = processed.train_frame.collect()
    _, sample_y = MultiHotDatasetWithPatientLookBack(
        train_frame,
        target_col="medication_multihot",
        n_diagnoses=len(processor.diagnoses_vocab.id_to_token),
        n_procedures=len(processor.procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )[0]
    output_size = sample_y.shape[0]

    ehr_adj = create_ehr_adjacency_matrix(train_frame)
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=medications_vocab,
        ddinter_path=DDI_PATH,
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}, DDI adj: {ddi_adj.shape}")

    # Run all models sequentially
    all_metrics: dict[str, dict] = {}

    all_metrics["mlp"]       = tune_mlp(processed, processor, device)
    all_metrics["foursdrug"] = tune_foursdrug(processed, processor, device)
    all_metrics["gamenet"]   = tune_gamenet(processed, processor, device, ehr_adj, ddi_adj)
    all_metrics["fastrx"]    = tune_fastrx(processed, processor, device, ehr_adj, ddi_adj)
    all_metrics["micron"]    = tune_micron(processed, processor, device, ddi_adj)

    # Write side-by-side comparison
    comparison_path = OUTPUT_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(all_metrics, indent=2))

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    metrics_to_show = [TARGET_METRIC, "F1", "PRAUC", "Binary DDI"]
    header = f"{'Model':<12}" + "".join(f"{m:>20}" for m in metrics_to_show)
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_metrics.items():
        row = f"{model_name:<12}" + "".join(
            f"{metrics.get(m, float('nan')):>20.4f}" for m in metrics_to_show
        )
        print(row)
    print(f"\nFull results: {OUTPUT_DIR.resolve()}/")
    print(f"Comparison:   {comparison_path.resolve()}")
