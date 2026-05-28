import polars as pl
import torch

from ehrdrec.mappings.code_to_id.vocab import Vocab

def create_ehr_adjacency_matrix(
    df: pl.DataFrame, 
    medication_col: str = "medication_multihot"
) -> torch.Tensor:
    med_matrix = torch.tensor(
        df[medication_col].to_list(),
        dtype=torch.float32
    )
    
    unique_combinations = torch.unique(med_matrix, dim=0)
    
    A_b = unique_combinations.T
    
    A_e = A_b @ A_b.T
    
    A_e.fill_diagonal_(0)
    
    return A_e

def create_ddi_adjacency_matrix(
    medications_vocab: Vocab,
    ddinter_path: str,
    n_medications: int,
    atc_level: int = 5,
) -> torch.Tensor:
    token_to_id = medications_vocab.token_to_id

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