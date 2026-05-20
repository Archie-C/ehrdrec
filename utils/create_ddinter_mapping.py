from collections import defaultdict

import polars as pl

folder_path = "data/ddinter2/"

files = [
    "ddinter_downloads_code_A.csv",
    "ddinter_downloads_code_B.csv",
    "ddinter_downloads_code_D.csv",
    "ddinter_downloads_code_H.csv",
    "ddinter_downloads_code_L.csv",
    "ddinter_downloads_code_P.csv",
    "ddinter_downloads_code_R.csv",
    "ddinter_downloads_code_V.csv",
    
]

id_to_codes = pl.read_csv("data/ddinter2/mapping/ddinter_atc_codes.csv")
stacked = pl.DataFrame()

for f in files:
    path = folder_path + f
    df = pl.read_csv(path)
    print(f"{f}: {df.height} rows, {df.width} columns")


    df = df.with_columns([
        pl.col("DDInterID_A").cast(pl.Utf8),
        pl.col("DDInterID_B").cast(pl.Utf8),
    ])

    id_to_codes = id_to_codes.with_columns([
        pl.col("drug_id").cast(pl.Utf8),
        pl.col("atc_code").cast(pl.Utf8),
    ])

    mapped = (
        df
        # join A ID to ATC
        .join(
            id_to_codes.rename({
                "drug_id": "DDInterID_A",
                "atc_code": "ATC_A",
            }),
            on="DDInterID_A",
            how="left",
        )
        # join B ID to ATC
        .join(
            id_to_codes.rename({
                "drug_id": "DDInterID_B",
                "atc_code": "ATC_B",
            }),
            on="DDInterID_B",
            how="left",
        )
    )
    
    stacked = pl.concat([stacked, mapped], how="vertical")

    print(stacked.height)
print(stacked.null_count())
stacked = stacked.drop_nulls(subset=["ATC_A", "ATC_B"])
ddi_lookup = (
    stacked
    .with_columns([
        pl.min_horizontal("ATC_A", "ATC_B").alias("ATC_A"),
        pl.max_horizontal("ATC_A", "ATC_B").alias("ATC_B"),
    ])
    .unique(subset=["ATC_A", "ATC_B"])
    .select(["ATC_A", "ATC_B", "Level"])
)
ddi_lookup.write_csv("data/ddinter2/mapping/ddinter_mapped_atc_codes.csv")

