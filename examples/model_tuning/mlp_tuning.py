import logging

import optuna
import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import MLP
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, Tuner
from ehrdrec.training.losses import BCELoss

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()
optuna.logging.set_verbosity(optuna.logging.WARNING)

ATC_LEVEL = 5

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)
    medications_vocab = processor.medications_vocab

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
    input_size = x.shape[0]
    output_size = y.shape[0]
    print(f"Input size: {input_size}, Output size: {output_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def make_metrics():
        return [
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

    def trial_fn(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader) -> Trainer:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        hidden_sizes = [hidden_size] * n_layers

        model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout=dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=BCELoss(),
            optimizer=optimizer,
            metrics=make_metrics(),
            target_metric="Jaccard",
            higher_is_better=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            epochs=20,
            trial=trial,
        )

    tuner = Tuner(trial_fn, n_trials=30, direction="maximize")
    study, best_results = tuner.tune(train_loader, val_loader)

    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation Jaccard: {study.best_value:.4f}")

    # Retrain with best params on train+val, then evaluate on test
    best_params = study.best_params
    hidden_sizes = [best_params["hidden_size"]] * best_params["n_layers"]
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=best_params["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=BCELoss(),
        optimizer=optimizer,
        metrics=make_metrics(),
        target_metric="Jaccard",
        higher_is_better=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=40,
    )
    results = trainer.fit()
    model.load_state_dict(results.best_model_state)

    evaluator = Evaluator(
        model=model,
        test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False),
        metrics=make_metrics(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_results = evaluator.run()
    print("\nTest evaluation results:")
    print(eval_results.test_metrics)
