import logging

import torch.nn as nn

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, ConsoleLogger, CheckpointLogger, CompositeLogger
from ehrdrec.training.losses import BCELoss
from ehrdrec.models import MLP
import torch

from torch.utils.data import DataLoader

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL = 5

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)
    medications_vocab = processor.medications_vocab
    
    print(processed_data.train_frame.columns)
    
    train_dataset = MultiHotDataset(processed_data.train_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    val_dataset = MultiHotDataset(processed_data.val_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    test_dataset = MultiHotDataset(processed_data.test_frame.collect(), target_col="medication_multihot", feature_cols=["diagnosis_multihot", "procedure_multihot"])
    x, y = train_dataset[0]
    output_size = y.shape[0]
    input_size = x.shape[0]
    print(f"Input size: {input_size}, Output size: {output_size}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = MLP(input_size=input_size, hidden_sizes=[14, 94, 267, 889], output_size=output_size, dropout=0.5)
    
    loss_fn = BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        )
    ]
    loggers = [
        ConsoleLogger(),
        CheckpointLogger(checkpoint_dir="mlp_checkpoints", keep_last=True),
    ]
    logger = CompositeLogger(loggers)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
        target_metric="Jaccard",
        higher_is_better=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=40,
        logger=logger,
    )
    results = trainer.fit()
    
    print("Training results:")
    print("Best epoch:", results.best_epoch)
    print("Best training metrics:", results.best_train_metrics)
    print("Best validation metrics:", results.best_val_metrics)
    
    model.load_state_dict(results.best_model_state)
    
    evaluator = Evaluator(
        model=model,
        test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False),
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_results = evaluator.run()
    print("Evaluation results:")
    print(eval_results.test_metrics)