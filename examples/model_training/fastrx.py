import logging

from ehrdrec.datasets.multi_hot import MultiHotDatasetWithPatientLookBack
from ehrdrec.datasets import collate_patient_visit_histories
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import Trainer, ConsoleLogger, CheckpointLogger, CompositeLogger
from ehrdrec.models import FastRx
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
import torch

from torch.utils.data import DataLoader

from ehrdrec.training.losses import BCELoss

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL = 5
LOOK_BACK = 3

if __name__ == "__main__":
    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")
    processor = MultiHotProcessor()
    processed_data = processor.process(data, minimum_admissions=2, atc_level=ATC_LEVEL, force_reload=True)
    
    diagnoses_vocab = processor.diagnoses_vocab
    procedures_vocab = processor.procedures_vocab
    medications_vocab = processor.medications_vocab
    
    train_dataset = MultiHotDatasetWithPatientLookBack(
        multi_hot_data_frame=processed_data.train_frame.collect(),
        target_col="medication_multihot",
        n_diagnoses=len(diagnoses_vocab.id_to_token),
        n_procedures=len(procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    val_dataset = MultiHotDatasetWithPatientLookBack(
        multi_hot_data_frame=processed_data.val_frame.collect(),
        target_col="medication_multihot",
        n_diagnoses=len(diagnoses_vocab.id_to_token),
        n_procedures=len(procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    test_dataset = MultiHotDatasetWithPatientLookBack(
        multi_hot_data_frame=processed_data.test_frame.collect(),
        target_col="medication_multihot",
        n_diagnoses=len(diagnoses_vocab.id_to_token),
        n_procedures=len(procedures_vocab.id_to_token),
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    
    collate_fn = collate_patient_visit_histories
    
    x, y = train_dataset[0]
    output_size = y.shape[0]
    print(f"Output size: {output_size}")
    
    # matrices
    ehr_adj_matrix = create_ehr_adjacency_matrix(processed_data.train_frame.collect())
    ddi_adj_matrix = create_ddi_adjacency_matrix(
        medications_vocab=medications_vocab,
        ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adjacency matrix shape: {ehr_adj_matrix.shape}")
    print(f"DDI adjacency matrix shape: {ddi_adj_matrix.shape}")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    model = FastRx(
        n_diagnoses=diagnoses_vocab.id_to_token.__len__(),
        n_procedures=procedures_vocab.id_to_token.__len__(),
        n_medications=output_size,
        medication_adjacency_matrix=ehr_adj_matrix,
        ddi_adjacency_matrix=ddi_adj_matrix,
        embedding_dim=256,
        embedding_dim_fastformer=128,
        dropout=0.5
    )
    
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
        CheckpointLogger(checkpoint_dir="fastrx_checkpoints", keep_last=True),
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
        epochs=100,
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
        test_loader=DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn),
        metrics=metrics,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_results = evaluator.run()
    print("Evaluation results:")
    print(eval_results.test_metrics)