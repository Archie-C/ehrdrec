import logging

import optuna
import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import FourSDrug
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, Tuner, TqdmLogger, TunerTqdmCallback
from ehrdrec.training.losses import BCELoss

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()
optuna.logging.set_verbosity(optuna.logging.WARNING)

ATC_LEVEL = 5
N_TRIALS = 30
TUNE_EPOCHS = 20
FINAL_EPOCHS = 40

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)
    medications_vocab = processor.medications_vocab
    n_diagnoses = len(processor.diagnoses_vocab.id_to_token)
    n_procedures = len(processor.procedures_vocab.id_to_token)

    dataset_kwargs = dict(target_col="medication_multihot", n_diagnoses=n_diagnoses, n_procedures=n_procedures)
    train_dataset = MultiHotDataset(processed_data.train_frame.collect(), **dataset_kwargs)
    val_dataset = MultiHotDataset(processed_data.val_frame.collect(), **dataset_kwargs)
    test_dataset = MultiHotDataset(processed_data.test_frame.collect(), **dataset_kwargs)

    x, y = train_dataset[0]
    input_size = x.shape[0]
    output_size = y.shape[0]
    print(f"Input size: {input_size}, Output size: {output_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
        emb_dim = trial.suggest_categorical("emb_dim", [32, 64, 128, 256])

        model = FourSDrug(
            num_symptoms=input_size,
            num_drugs=output_size,
            emb_dim=emb_dim,
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
            device="cuda" if torch.cuda.is_available() else "cpu",
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
    model = FourSDrug(
        num_symptoms=input_size,
        num_drugs=output_size,
        emb_dim=best_params["emb_dim"],
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=FINAL_EPOCHS,
        logger=TqdmLogger(epochs=FINAL_EPOCHS, metrics=["Jaccard"], desc="Final training"),
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    evaluator = Evaluator(
        model=model,
        test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False),
        metrics=make_full_metrics(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_results = evaluator.run()
    print("\nTest evaluation results:")
    print(eval_results.test_metrics)
