import torch
import polars as pl
from ehrdrec.mappings.code_to_id.vocab import Vocab
from ehrdrec.metrics.base import Metric

# TODO: Remove UNK when scoring, double check this is all correct too.
class BinaryDDI(Metric):
    def __init__(
        self,
        *,
        medications_vocab: Vocab,
        ddinter_path: str,
        n_medications: int,
        atc_level: int = 3,
        name: str = "Binary DDI",
        threshold: float = 0.5,
        from_logits: bool = True,
        device: torch.device | str = "cuda",
    ):
        super().__init__(name)

        self.medications_vocab = medications_vocab
        self.threshold = threshold
        self.from_logits = from_logits
        self.device = torch.device(device)

        self.ddi_adj = self.build_ddi_adj(ddinter_path, n_medications, atc_level=atc_level).to(self.device)

        self.upper_mask = torch.triu(
            torch.ones((n_medications, n_medications), dtype=torch.bool, device=self.device),
            diagonal=1,
        )

        self.total_ddi = torch.tensor(0.0, device=self.device)
        self.total_pairs = torch.tensor(0.0, device=self.device)

        
    def build_ddi_adj(
        self,
        ddinter_path: str,
        n_medications: int,
        atc_level: int = 3,
    ) -> torch.Tensor:
        token_to_id = self.medications_vocab.token_to_id

        severity_rank = {
            "Unknown": 1, # TODO: TUNE - should we include them? If so, do we be pessimistic and treat them as "Major" or optimistic and treat them as "Minor"?
            "Minor": 1,
            "Moderate": 2,
            "Major": 3,
        }

        def truncate_atc(code: str) -> str:
            if atc_level is None:
                return code

            # ATC level examples:
            # 1: A
            # 2: A10
            # 3: A10B
            # 4: A10BA
            # 5: A10BA02
            level_lengths = {
                1: 1,
                2: 3,
                3: 4,
                4: 5,
                5: 7,
            }

            return code[:level_lengths[atc_level]]

        df = (
            pl.read_csv(ddinter_path)
            .select(["ATC_A", "ATC_B", "Level"])
            .drop_nulls()
            .with_columns([
                pl.col("ATC_A").map_elements(truncate_atc, return_dtype=pl.Utf8),
                pl.col("ATC_B").map_elements(truncate_atc, return_dtype=pl.Utf8),
                pl.col("Level").replace(severity_rank).cast(pl.Int64).alias("severity"),
            ])
            .with_columns([
                pl.min_horizontal("ATC_A", "ATC_B").alias("ATC_1"),
                pl.max_horizontal("ATC_A", "ATC_B").alias("ATC_2"),
            ])
            .filter(pl.col("ATC_1") != pl.col("ATC_2"))
            .sort("severity", descending=True)
            .unique(subset=["ATC_1", "ATC_2"], keep="first")
        )

        adj = torch.zeros((n_medications, n_medications), dtype=torch.bool)

        for atc_a, atc_b in df.select(["ATC_1", "ATC_2"]).iter_rows():
            id_a = token_to_id.get(atc_a)
            id_b = token_to_id.get(atc_b)

            if (
                id_a is None
                or id_b is None
                or id_a == id_b
                or id_a >= n_medications
                or id_b >= n_medications
            ):
                continue

            adj[id_a, id_b] = True
            adj[id_b, id_a] = True

        return adj

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        x = outputs.detach().to(self.device)

        if self.from_logits:
            x = x.sigmoid()

        preds = x >= self.threshold

        # [batch, meds, meds]
        pair_mask = preds.unsqueeze(2) & preds.unsqueeze(1)

        valid_pairs = pair_mask & self.upper_mask
        ddi_pairs = valid_pairs & self.ddi_adj

        self.total_ddi += ddi_pairs.sum()
        self.total_pairs += valid_pairs.sum()

    def compute(self) -> float:
        if self.total_pairs.item() == 0:
            return 0.0

        return (self.total_ddi / self.total_pairs).item()

    def reset(self) -> None:
        self.total_ddi.zero_()
        self.total_pairs.zero_()
        

class HighSeverityBinaryDDI(BinaryDDI):
    def __init__(
        self,
        *,
        medications_vocab: Vocab,
        ddinter_path: str,
        n_medications: int,
        atc_level: int = 3,
        name: str = "High Severity Binary DDI",
        threshold: float = 0.5,
        from_logits: bool = True,
        device: torch.device | str = "cuda",
    ):
        super().__init__(
            medications_vocab=medications_vocab,
            ddinter_path=ddinter_path,
            n_medications=n_medications,
            atc_level=atc_level,
            name=name,
            threshold=threshold,
            from_logits=from_logits,
            device=device,
        )
        
    def build_ddi_adj(
        self,
        ddinter_path: str,
        n_medications: int,
        atc_level: int = 3,
    ) -> torch.Tensor:
        token_to_id = self.medications_vocab.token_to_id

        severity_rank = {
            "Unknown": 1, # TODO: TUNE - should we include them? If so, do we be pessimistic and treat them as "Major" or optimistic and treat them as "Minor"?
            "Minor": 1,
            "Moderate": 2,
            "Major": 3,
        }

        def truncate_atc(code: str) -> str:
            if atc_level is None:
                return code

            # ATC level examples:
            # 1: A
            # 2: A10
            # 3: A10B
            # 4: A10BA
            # 5: A10BA02
            level_lengths = {
                1: 1,
                2: 3,
                3: 4,
                4: 5,
                5: 7,
            }

            return code[:level_lengths[atc_level]]

        df = (
            pl.read_csv(ddinter_path)
            .select(["ATC_A", "ATC_B", "Level"])
            .drop_nulls()
            .with_columns([
                pl.col("ATC_A").map_elements(truncate_atc, return_dtype=pl.Utf8),
                pl.col("ATC_B").map_elements(truncate_atc, return_dtype=pl.Utf8),
                pl.col("Level").replace(severity_rank).cast(pl.Int64).alias("severity"),
            ])
            .with_columns([
                pl.min_horizontal("ATC_A", "ATC_B").alias("ATC_1"),
                pl.max_horizontal("ATC_A", "ATC_B").alias("ATC_2"),
            ])
            .filter((pl.col("ATC_1") != pl.col("ATC_2")) & (pl.col("severity") >= severity_rank["Major"]))
            .sort("severity", descending=True)
            .unique(subset=["ATC_1", "ATC_2"], keep="first")
        )

        adj = torch.zeros((n_medications, n_medications), dtype=torch.bool)

        for atc_a, atc_b in df.select(["ATC_1", "ATC_2"]).iter_rows():
            id_a = token_to_id.get(atc_a)
            id_b = token_to_id.get(atc_b)

            if (
                id_a is None
                or id_b is None
                or id_a == id_b
                or id_a >= n_medications
                or id_b >= n_medications
            ):
                continue

            adj[id_a, id_b] = True
            adj[id_b, id_a] = True 
        
        return adj