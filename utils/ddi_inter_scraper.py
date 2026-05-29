import requests
from bs4 import BeautifulSoup
import polars as pl
from pathlib import Path
import csv
import time

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

unique_ids = set()
for f in files:
    path = Path(folder_path) / f
    df = pl.read_csv(path)
    unique_ids = (
        set(df["DDInterID_A"].drop_nulls().unique().to_list())
        | set(df["DDInterID_B"].drop_nulls().unique().to_list())
        | unique_ids
    )

print(f"Total unique DDInter IDs: {len(unique_ids)}")

results = []  # list of (drug_id, atc_code)
no_atc = []   # drug IDs with no ATC code found
errors = []   # drug IDs that failed to fetch

for i, drug_id in enumerate(sorted(unique_ids)):
    try:
        url = f"https://ddinter2.scbdd.com/server/drug-detail/{drug_id}/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        found = False
        for row in soup.find_all("tr"):
            key = row.find("td", class_="key")
            if key and "ATC Classification" in key.text:
                value_td = row.find("td", class_="value")
                if value_td is None:          # skip if no value cell
                    break
                atc_spans = value_td.find_all("span", class_="badge")
                atc_codes = [span.text.strip() for span in atc_spans]
                if atc_codes:
                    print(f"[{i+1}] {drug_id}: {atc_codes}")
                    for code in atc_codes:
                        results.append((drug_id, code))
                    found = True
                break

        if not found:
            no_atc.append(drug_id)

        time.sleep(0.3)

    except Exception as e:
        print(f"[{i+1}] {drug_id}: ERROR - {e}")
        errors.append(drug_id)
        time.sleep(1)

# Save to CSV
output_file = "ddinter_atc_codes.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drug_id", "atc_code"])
    writer.writerows(results)

print(f"\n✅ Done!")
print(f"   Drugs with ATC codes : {len(set(r[0] for r in results))}")
print(f"   Total ATC mappings   : {len(results)}")
print(f"   No ATC code found    : {len(no_atc)}")
print(f"   Errors               : {len(errors)}")
print(f"   Output saved to      : {output_file}")

if errors:
    print(f"\n⚠️  Failed IDs: {errors}")