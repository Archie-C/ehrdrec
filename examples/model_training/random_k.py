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
SEED = 42


class RandomKMedicationPredictor(nn.Module):
    def __init__(
        self,
        *,
        n_medications: int,
        k: int,
        seed: int,
        high_logit: float = 10.0,
        low_logit: float = -10.0,
        first_candidate_id: int = 2,
    ) -> None:
        super().__init__()
        if k < 0:
            raise ValueError("k must be non-negative.")
        if first_candidate_id >= n_medications and k > 0:
            raise ValueError("No medication ids are available to sample.")

        self.n_medications = n_medications
        self.k = min(k, n_medications - first_candidate_id)
        self.high_logit = high_logit
        self.low_logit = low_logit
        self.first_candidate_id = first_candidate_id
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, x):
        batch_size = x.shape[0] if torch.is_tensor(x) else next(iter(x.values())).shape[0]
        device = x.device if torch.is_tensor(x) else next(iter(x.values())).device

        logits = torch.full(
            (batch_size, self.n_medications),
            self.low_logit,
            device=device,
        )
        if self.k == 0:
            return logits

        candidate_count = self.n_medications - self.first_candidate_id
        for row_idx in range(batch_size):
            sampled = torch.randperm(candidate_count, generator=self.generator)[: self.k]
            sampled = sampled.to(device) + self.first_candidate_id
            logits[row_idx, sampled] = self.high_logit

        return logits


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
    val_dataset = MultiHotDataset(processed_data.val_frame.collect(), **dataset_kwargs)
    test_dataset = MultiHotDataset(processed_data.test_frame.collect(), **dataset_kwargs)

    _, sample_target = val_dataset[0]
    n_medications = sample_target.shape[0]
    print(f"Random-{K} baseline over {n_medications} medications")

    model = RandomKMedicationPredictor(
        n_medications=n_medications,
        k=K,
        seed=SEED,
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
