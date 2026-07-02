import logging

import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import MRDTRDataset, build_mrdtr_graph, collate_mrdtr_examples
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models import MRDTR
from ehrdrec.models.utils import create_ehr_adjacency_matrix, create_ddi_adjacency_matrix
from ehrdrec.processing import SetSequenceProcessor
from ehrdrec.training import Trainer, CheckpointLogger, CompositeLogger, TqdmLogger
from ehrdrec.training.losses import BCELoss

logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

ATC_LEVEL      = 5
BATCH_SIZE     = 1
EPOCHS         = 40
LR             = 1e-3
SEED           = 42
HOP_NUM        = 2


class MRDTRTrainingAdapter(torch.nn.Module):
    def __init__(self, model: MRDTR):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(
            hop_node_indices=batch.hop_node_indices,
            hop_node_temporal_features=batch.hop_node_temporal_features,
            central_node_temporal_feature=batch.central_node_temporal_feature,
            diagnosis_code_lists=batch.diagnosis_code_lists,
            procedure_code_lists=batch.procedure_code_lists,
        )


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4")

    processor = SetSequenceProcessor()
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
        n_medications=processor.medications_vocab.vocab_size,
        time_col="admission_time",
        hop_num=HOP_NUM,
        dtype=torch.float32,
    )

    train_graph = build_mrdtr_graph(train_frame)
    val_graph = build_mrdtr_graph(val_frame)
    test_graph = build_mrdtr_graph(test_frame)

    train_dataset = MRDTRDataset(train_frame, graph=train_graph, **dataset_kwargs)
    val_dataset   = MRDTRDataset(val_frame,   graph=val_graph,   **dataset_kwargs)
    test_dataset  = MRDTRDataset(test_frame,  graph=test_graph,  **dataset_kwargs)

    output_size = processor.medications_vocab.vocab_size
    print(f"Output size: {output_size}")

    ehr_adj = create_ehr_adjacency_matrix(
        train_frame,
        medication_col="atc_ids",
        n_medications=output_size,
    )
    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path="data/ddinter2/mapping/ddinter_mapped_atc_codes.csv",
        n_medications=output_size,
        atc_level=ATC_LEVEL,
    )
    print(f"EHR adj: {ehr_adj.shape}, DDI adj: {ddi_adj.shape}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_mrdtr_examples)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mrdtr_examples)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_mrdtr_examples)

    mrdtr = MRDTR(
        n_diagnoses = len(processor.diagnoses_vocab.id_to_token),
        n_procedures = len(processor.procedures_vocab.id_to_token),
        n_medications = output_size,
        n_patients = len(train_graph["patient"]),
        embedding_dim = 128,
        embedding_dropout = 0.1,
        temporal_attention_dropout = 0.1,
        temporal_information_importance = 0.5,
        ehr_adjacency_matrix = ehr_adj,
        ddi_adjacency_matrix = ddi_adj,
        device=device,
        hop_num = HOP_NUM,
        temporal_feature_dim = 1
    )

    model = MRDTRTrainingAdapter(mrdtr)

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
        TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="MRDTR"),
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
