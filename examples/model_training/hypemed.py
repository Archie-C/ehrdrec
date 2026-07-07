import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.datasets import HypeMedDataset, collate_hypemed_examples
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import BinaryDDI, F1, Jaccard, PRAUC
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.HypeMed import HypeMed, HypeMedPretrainer, construct_graphs
from ehrdrec.models.utils import create_ddi_adjacency_matrix
from ehrdrec.processing import SetSequenceProcessor
from ehrdrec.training import CheckpointLogger, CompositeLogger, TqdmLogger, Trainer


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-3
EMB_DIM = 128
N_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
TOP_N = 10
LOOK_BACK = None
MIN_VISITS = 2
SEED = 42

PRETRAIN_EPOCHS = 10
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-5
DROP_INCIDENCE_RATE = 0.2
DROP_FEATURE_RATE = 0.2
TAU_N = 0.5
TAU_G = 0.5
TAU_M = 0.5
TAU_C = 0.5
PRETRAIN_BATCH_SIZE_1 = 1024
PRETRAIN_BATCH_SIZE_2 = 4096
W_G = 1.0
W_M = 1.0
PROJECTION_DIM = 128

DDI_WEIGHT = 0.05
SSL_WEIGHT = 0.01
ORTHOGONALITY_WEIGHT = 0.01


class HypeMedLoss(nn.Module):
    def __init__(
        self,
        *,
        ddi_weight: float = 0.05,
        ssl_weight: float = 0.01,
        orthogonality_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.ddi_weight = ddi_weight
        self.ssl_weight = ssl_weight
        self.orthogonality_weight = orthogonality_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, model_output=None, features=None, losses=None, **kwargs):
        loss = self.bce_loss(predictions, targets)
        if losses is None:
            return loss

        if "ddi_loss" in losses:
            loss = loss + self.ddi_weight * losses["ddi_loss"].to(predictions.device)
        if "ssl_loss" in losses:
            loss = loss + self.ssl_weight * losses["ssl_loss"].to(predictions.device)
        if "orthogonality_loss" in losses:
            loss = loss + self.orthogonality_weight * losses["orthogonality_loss"].to(predictions.device)
        return loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_metrics(medications_vocab, n_medications: int, device: torch.device) -> list:
    return [
        Jaccard(),
        F1(),
        PRAUC(),
        BinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
            device=device,
        ),
        HighSeverityBinaryDDI(
            medications_vocab=medications_vocab,
            ddinter_path=DDINTER_PATH,
            n_medications=n_medications,
            atc_level=ATC_LEVEL,
            device=device,
        ),
    ]


def build_hypemed_model(
    *,
    n_diagnoses: int,
    n_procedures: int,
    n_medications: int,
    n_ehr_edges: int,
    ddi_adj: torch.Tensor,
    x_hat: dict[str, torch.Tensor],
    e_mem: dict[str, torch.Tensor],
    device: torch.device,
) -> HypeMed:
    return HypeMed(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        embedding_dim=EMB_DIM,
        number_of_heads=N_HEADS,
        number_of_ehr_edges=n_ehr_edges,
        top_n=min(TOP_N, n_ehr_edges),
        device=device,
        X_hat=x_hat,
        E_mem=e_mem,
        ddi_adjacency_matrix=ddi_adj,
        dropout=DROPOUT,
    ).to(device)


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = MIMIC3Loader()
    data = loader.load(MIMIC3_PATH)

    processor = SetSequenceProcessor()
    processed = processor.process(
        data,
        minimum_admissions=MIN_VISITS,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )

    sort_cols = ["patient_id", "admission_time"]
    train_frame = processed.train_frame.collect().sort(sort_cols).with_row_index("hypemed_edge_id")
    val_frame = processed.val_frame.collect().sort(sort_cols)
    test_frame = processed.test_frame.collect().sort(sort_cols)

    n_diagnoses = processor.diagnoses_vocab.vocab_size
    n_procedures = processor.procedures_vocab.vocab_size
    n_medications = processor.medications_vocab.vocab_size
    n_ehr_edges = train_frame.height

    print(
        "Vocab sizes:",
        f"diagnoses={n_diagnoses}",
        f"procedures={n_procedures}",
        f"medications={n_medications}",
        f"ehr_edges={n_ehr_edges}",
    )

    graph_dict = construct_graphs(
        train_frame,
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        medication_col="atc_ids",
    )

    pretrainer = HypeMedPretrainer(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        num_edges=n_ehr_edges,
        adjacency_dict=graph_dict,
        pretrain_epochs=PRETRAIN_EPOCHS,
        pretrain_learning_rate=PRETRAIN_LR,
        pretrain_weight_decay=PRETRAIN_WEIGHT_DECAY,
        drop_incidence_rate=DROP_INCIDENCE_RATE,
        drop_feature_rate=DROP_FEATURE_RATE,
        tau_n=TAU_N,
        tau_g=TAU_G,
        tau_m=TAU_M,
        tau_c=TAU_C,
        batch_size_1=PRETRAIN_BATCH_SIZE_1,
        batch_size_2=PRETRAIN_BATCH_SIZE_2,
        w_g=W_G,
        w_m=W_M,
        embedding_dim=EMB_DIM,
        projection_dim=PROJECTION_DIM,
        number_of_heads=N_HEADS,
        dropout=DROPOUT,
        number_of_layers=NUM_LAYERS,
        device=device,
        cache_dir="cache/hypemed_pretrain",
    )
    pretrain_history = pretrainer.pretrain()
    print("Pretraining losses:", {k: v[-1] for k, v in pretrain_history.items() if v})
    x_hat, e_mem = pretrainer.get_encoded_embeddings()
    x_hat = {k: v.detach().to(device) for k, v in x_hat.items()}
    e_mem = {k: v.detach().to(device) for k, v in e_mem.items()}

    dataset_kwargs = dict(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        patient_id_col="patient_id",
        time_col="admission_time",
        diagnosis_col="diagnosis_ids",
        procedure_col="procedure_ids",
        medication_col="atc_ids",
        medication_is_multihot=False,
        min_visits=MIN_VISITS,
        sample_all_visits=True,
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    train_dataset = HypeMedDataset(train_frame, edge_id_col="hypemed_edge_id", **dataset_kwargs)
    val_dataset = HypeMedDataset(val_frame, **dataset_kwargs)
    test_dataset = HypeMedDataset(test_frame, **dataset_kwargs)

    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=n_medications,
        atc_level=ATC_LEVEL,
    )
    print(f"DDI adj: {ddi_adj.shape}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_hypemed_examples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_hypemed_examples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_hypemed_examples,
    )

    model = build_hypemed_model(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        n_ehr_edges=n_ehr_edges,
        ddi_adj=ddi_adj,
        x_hat=x_hat,
        e_mem=e_mem,
        device=device,
    )

    logger = CompositeLogger([
        TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="HypeMed"),
        CheckpointLogger(checkpoint_dir="checkpoints/hypemed", keep_last=True),
    ])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=HypeMedLoss(
            ddi_weight=DDI_WEIGHT,
            ssl_weight=SSL_WEIGHT,
            orthogonality_weight=ORTHOGONALITY_WEIGHT,
        ),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        metrics=make_metrics(processor.medications_vocab, n_medications, device),
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
        metrics=make_metrics(processor.medications_vocab, n_medications, device),
        device=device,
    ).run()
    print("Test metrics:", eval_results.test_metrics)
