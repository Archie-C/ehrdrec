from ehrdrec.loading.mimic_iii import MIMIC3Loader
from ehrdrec.mappings import NDCATCMapper

def check_table():
    import sqlite3

    conn = sqlite3.connect("data/mappings/ndc_atc_mapping.sqlite")

    for table in ["ndc_to_rxcui", "rxcui_to_atc", "ndc_to_atc"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(table, count)

    conn.close()

def top_20_atc_codes():
    import sqlite3

    conn = sqlite3.connect("data/mappings/ndc_atc_mapping.sqlite")

    rows = conn.execute(
        """
        SELECT ndc, raw_ndc, drug_rxcui, ingredient_rxcui, atc_code, atc_name, match_type
        FROM ndc_to_atc
        LIMIT 20
        """
    ).fetchall()

    for row in rows:
        print(row)

    conn.close()

def main():
    mapper = NDCATCMapper.from_file("data/mappings/ndc_atc_mapping.sqlite")

    loader = MIMIC3Loader()
    data = loader.load("/home/cararc/data/mimic-iii-1.4", force_reload=True)

    rows = data.frame.select("medications").collect()

    total_ndcs = 0
    matched_ndcs = 0
    missing_ndcs = set()
    matched_examples = []

    for row in rows.iter_rows(named=True):
        meds = row["medications"] or []
        for med in meds:
            ndc = med.get("NDC") if isinstance(med, dict) else med["NDC"]
            
            if ndc is None or str(ndc).strip() in {"", "0"}:
                continue
            
            total_ndcs += 1

            result = mapper.ndc_to_atc(str(ndc))

            if result and result.atc_codes:
                matched_ndcs += 1

                if len(matched_examples) < 20:
                    matched_examples.append((ndc, result.atc_codes))
            else:
                missing_ndcs.add(str(ndc))

    print(f"Total NDC entries checked: {total_ndcs}")
    print(f"Matched NDC entries: {matched_ndcs}")
    print(f"Missing NDC entries: {total_ndcs - matched_ndcs}")
    print(f"Coverage: {matched_ndcs / total_ndcs:.2%}" if total_ndcs else "No NDCs found")

    print("\nMatched examples:")
    for ndc, atcs in matched_examples:
        print(f"{ndc} -> {atcs}")

    print("\nMissing examples:")
    for ndc in list(missing_ndcs)[:20]:
        print(ndc)

if __name__ == "__main__":
    main()