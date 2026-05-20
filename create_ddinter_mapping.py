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

id_to_codes = pl.read_csv("ddinter_atc_codes.csv")
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
stacked.drop_nulls(subset=["ATC_A", "ATC_B"]).write_csv("ddinter_mapped_atc_codes_nonnull.csv")
stacked.write_csv("ddinter_mapped_atc_codes.csv")

null_a = (
    stacked
    .filter(pl.col("ATC_A").is_null())
    .select("DDInterID_A", "Drug_A")
    .unique()
)
null_b = (
    stacked
    .filter(pl.col("ATC_B").is_null())
    .select("DDInterID_B", "Drug_B")
    .unique()
)

combined = set(zip(null_a.get_column("DDInterID_A"), null_a.get_column("Drug_A"))) | set(zip(null_b.get_column("DDInterID_B"), null_b.get_column("Drug_B")))

if not combined:
    print("No unmapped IDs found.")
else:
    combined_list = sorted(combined, key=lambda x: (x[0] or "", x[1] or ""))
    print(f"Found {len(combined_list)} unmapped ID(s):")
    for ddid, drug in combined_list:
        print(f"- {ddid}: {drug}")