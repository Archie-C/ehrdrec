import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ehrdrec.datasets.multi_hot import MultiHotDataset
from ehrdrec.evaluation import Evaluator
from ehrdrec.loading import MIMIC3Loader
from ehrdrec.metrics import F1, Jaccard, PRAUC, BinaryDDI
from ehrdrec.metrics.ddi import HighSeverityBinaryDDI
from ehrdrec.processing import MultiHotProcessor


logging.getLogger("ehrdrec").setLevel(logging.INFO)
logging.basicConfig()

MIMIC3_PATH = "/home/cararc/data/mimic-iii-1.4"
DDINTER_PATH = "data/ddinter2/mapping/ddinter_mapped_atc_codes.csv"

ATC_LEVEL = 5
BATCH_SIZE = 256
K = 20


class PopularKMedicationPredictor(nn.Module):
    def __init__(
        self,
        *,
        medication_ids: torch.Tensor,
        n_medications: int,
        high_logit: float = 10.0,
        low_logit: float = -10.0,
    ) -> None:
        super().__init__()
        self.n_medications = n_medications
        self.high_logit = high_logit
        self.low_logit = low_logit
        self.register_buffer("medication_ids", medication_ids.long())

    def forward(self, x):
        batch_size = x.shape[0] if torch.is_tensor(x) else next(iter(x.values())).shape[0]
        device = x.device if torch.is_tensor(x) else next(iter(x.values())).device

        logits = torch.full(
            (batch_size, self.n_medications),
            self.low_logit,
            device=device,
        )
        if self.medication_ids.numel() > 0:
            logits[:, self.medication_ids.to(device)] = self.high_logit

        return logits


def top_k_medications(
    dataset: MultiHotDataset,
    *,
    k: int,
    first_candidate_id: int = 2,
) -> torch.Tensor:
    counts = torch.stack(
        [
            torch.tensor(row, dtype=torch.float32)
            for row in dataset.data_frame[dataset.target_col].to_list()
        ]
    ).sum(dim=0)

    counts[:first_candidate_id] = -1
    candidate_count = max(0, counts.numel() - first_candidate_id)
    k = min(k, candidate_count)
    if k == 0:
        return torch.empty(0, dtype=torch.long)

    return torch.topk(counts, k=k).indices.sort().values


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = MIMIC3Loader()
    data = loader.load(MIMIC3_PATH)

    processor = MultiHotProcessor()
    processed_data = processor.process(
        data,
        minimum_admissions=2,
        atc_level=ATC_LEVEL,
        force_reload=False,
    )

    medications_vocab = processor.medications_vocab
    n_diagnoses = processor.diagnoses_vocab.vocab_size
    n_procedures = processor.procedures_vocab.vocab_size

    dataset_kwargs = dict(
        target_col="medication_multihot",
        n_diagnoses=n_diagnoses,
        n_procedures=n_procedures,
    )
    train_dataset = MultiHotDataset(processed_data.train_frame.collect(), **dataset_kwargs)
    val_dataset = MultiHotDataset(processed_data.val_frame.collect(), **dataset_kwargs)
    test_dataset = MultiHotDataset(processed_data.test_frame.collect(), **dataset_kwargs)

    _, sample_target = train_dataset[0]
    n_medications = sample_target.shape[0]
    medication_ids = top_k_medications(train_dataset, k=K)
    print(f"Popular-{len(medication_ids)} baseline over {n_medications} medications")
    print("Medication ids:", medication_ids.tolist())

    model = PopularKMedicationPredictor(
        medication_ids=medication_ids,
        n_medications=n_medications,
    ).to(device)

    metrics = [
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

    val_results = Evaluator(
        model=model,
        test_loader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        metrics=metrics,
        device=device,
    ).run()
    print("Validation metrics:", val_results.test_metrics)

    test_results = Evaluator(
        model=model,
        test_loader=DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
        metrics=metrics,
        device=device,
    ).run()
    print("Test metrics:", test_results.test_metrics)
