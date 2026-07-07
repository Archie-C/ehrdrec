import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from ehrdrec.datasets import MultiHotDatasetWithPatientLookBack, collate_patient_visit_histories
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import BinaryDDI, F1, Jaccard, PRAUC
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.models.torch.original.rpnet import RPNet
from ehrdrec.models.utils import create_ddi_adjacency_matrix
from ehrdrec.processing import MultiHotProcessor
from ehrdrec.training import CheckpointLogger, CompositeLogger, TqdmLogger, Trainer
from ehrdrec.training.losses import OriginalGAMENetLoss


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
LOOK_BACK = 3
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
EMB_DIM = 128
ENCODER_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.1
PATIENT_SEPARATE = True
DDI_WEIGHT = 0.05
SEED = 42

PRETRAIN = True
PRETRAIN_EPOCHS = 20
PRETRAIN_LR = 1e-3
PRETRAIN_MASK_RATE = 0.15
PRETRAIN_CONTRASTIVE_WEIGHT = 0.1
PRETRAIN_TEMPERATURE = 0.2


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


def build_rpnet(
    *,
    n_diagnoses: int,
    n_procedures: int,
    n_medications: int,
    ddi_adj: torch.Tensor,
    device: torch.device,
) -> RPNet:
    return RPNet(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        embedding_dim=EMB_DIM,
        encoder_layers=ENCODER_LAYERS,
        number_of_heads=N_HEADS,
        dropout=DROPOUT,
        patient_separate=PATIENT_SEPARATE,
        ddi_adjacency_matrix=ddi_adj,
        device=device,
    ).to(device)


class RPNetPretrainingHead(nn.Module):
    def __init__(self, rpnet: RPNet) -> None:
        super().__init__()
        self.rpnet = rpnet
        self.diagnosis_head = nn.Linear(rpnet.embedding_dim, rpnet.n_diagnoses)
        self.procedure_head = nn.Linear(rpnet.embedding_dim, rpnet.n_procedures)

    def forward(self, features: dict[str, torch.Tensor], mask_rate: float) -> dict[str, torch.Tensor]:
        diagnoses = self._drop_codes(features["diagnoses"], mask_rate)
        procedures = self._drop_codes(features["procedures"], mask_rate)
        lengths = features["lengths"]

        visit_representations = self.rpnet.patient_encoder(diagnoses, procedures, lengths)
        return {
            "visit_representations": visit_representations,
            "diagnosis_logits": self.diagnosis_head(visit_representations),
            "procedure_logits": self.procedure_head(visit_representations),
        }

    @staticmethod
    def _drop_codes(values: torch.Tensor, mask_rate: float) -> torch.Tensor:
        if mask_rate <= 0:
            return values
        keep = torch.rand_like(values) >= mask_rate
        return values * keep.to(values.dtype)


def pretrain_rpnet(
    model: RPNet,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    mask_rate: float,
    contrastive_weight: float,
    temperature: float,
) -> list[dict[str, float]]:
    pretraining_model = RPNetPretrainingHead(model).to(device)
    optimizer = torch.optim.Adam(pretraining_model.parameters(), lr=learning_rate)
    history = []

    for epoch in trange(1, epochs + 1, desc="RPNet pretraining"):
        pretraining_model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_contrastive = 0.0
        total_visits = 0

        for features, _ in train_loader:
            features = {key: value.to(device) for key, value in features.items()}
            optimizer.zero_grad(set_to_none=True)

            output = pretraining_model(features, mask_rate=mask_rate)
            valid_visit_mask = _valid_visit_mask(features["lengths"], features["diagnoses"].size(1))

            reconstruction_loss = _masked_bce(
                output["diagnosis_logits"],
                features["diagnoses"],
                valid_visit_mask,
            ) + _masked_bce(
                output["procedure_logits"],
                features["procedures"],
                valid_visit_mask,
            )
            contrastive_loss = _next_visit_contrastive_loss(
                output["visit_representations"],
                features["lengths"],
                temperature=temperature,
            )
            loss = reconstruction_loss + contrastive_weight * contrastive_loss

            loss.backward()
            optimizer.step()

            visit_count = int(valid_visit_mask.sum().item())
            total_loss += loss.item() * visit_count
            total_reconstruction += reconstruction_loss.item() * visit_count
            total_contrastive += contrastive_loss.item() * visit_count
            total_visits += visit_count

        divisor = max(total_visits, 1)
        epoch_history = {
            "epoch": epoch,
            "loss": total_loss / divisor,
            "reconstruction_loss": total_reconstruction / divisor,
            "contrastive_loss": total_contrastive / divisor,
        }
        history.append(epoch_history)
        print(
            f"Pretrain epoch {epoch}: "
            f"loss={epoch_history['loss']:.4f}, "
            f"recon={epoch_history['reconstruction_loss']:.4f}, "
            f"contrastive={epoch_history['contrastive_loss']:.4f}"
        )

    return history


def _valid_visit_mask(lengths: torch.Tensor, max_visits: int) -> torch.Tensor:
    positions = torch.arange(max_visits, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, valid_visit_mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
    loss = loss * valid_visit_mask.to(loss.dtype)
    return loss.sum() / valid_visit_mask.sum().clamp_min(1)


def _next_visit_contrastive_loss(
    visit_representations: torch.Tensor,
    lengths: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    current_reps = []
    next_reps = []
    for batch_idx, length in enumerate(lengths.tolist()):
        if length < 2:
            continue
        current_reps.append(visit_representations[batch_idx, : length - 1])
        next_reps.append(visit_representations[batch_idx, 1:length])

    if not current_reps:
        return visit_representations.sum() * 0

    current = F.normalize(torch.cat(current_reps, dim=0), dim=-1)
    next_ = F.normalize(torch.cat(next_reps, dim=0), dim=-1)
    logits = current.matmul(next_.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    set_seed(SEED)
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

    n_diagnoses = processor.diagnoses_vocab.vocab_size
    n_procedures = processor.procedures_vocab.vocab_size
    n_medications = processor.medications_vocab.vocab_size
    print(
        "Vocab sizes:",
        f"diagnoses={n_diagnoses}",
        f"procedures={n_procedures}",
        f"medications={n_medications}",
    )

    dataset_kwargs = dict(
        target_col="medication_multihot",
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        patient_id_col="patient_id",
        time_col="admission_time",
        look_back=LOOK_BACK,
        dtype=torch.float32,
    )
    train_dataset = MultiHotDatasetWithPatientLookBack(train_frame, **dataset_kwargs)
    val_dataset = MultiHotDatasetWithPatientLookBack(val_frame, **dataset_kwargs)
    test_dataset = MultiHotDatasetWithPatientLookBack(test_frame, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_patient_visit_histories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_patient_visit_histories,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_patient_visit_histories,
    )

    ddi_adj = create_ddi_adjacency_matrix(
        medications_vocab=processor.medications_vocab,
        ddinter_path=DDINTER_PATH,
        n_medications=n_medications,
        atc_level=ATC_LEVEL,
    )
    print(f"DDI adj: {ddi_adj.shape}")

    model = build_rpnet(
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
        n_medications=n_medications,
        ddi_adj=ddi_adj,
        device=device,
    )

    if PRETRAIN and PRETRAIN_EPOCHS > 0:
        pretrain_history = pretrain_rpnet(
            model,
            train_loader,
            device=device,
            epochs=PRETRAIN_EPOCHS,
            learning_rate=PRETRAIN_LR,
            mask_rate=PRETRAIN_MASK_RATE,
            contrastive_weight=PRETRAIN_CONTRASTIVE_WEIGHT,
            temperature=PRETRAIN_TEMPERATURE,
        )
        print("Final pretraining losses:", pretrain_history[-1])

    logger = CompositeLogger(
        [
            TqdmLogger(epochs=EPOCHS, metrics=["Jaccard"], desc="RPNet"),
            CheckpointLogger(
                checkpoint_dir="checkpoints/rpnet",
                keep_last=True,
            ),
        ]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=OriginalGAMENetLoss(ddi_weight=DDI_WEIGHT),
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
