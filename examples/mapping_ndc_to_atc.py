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
    check_table()
    # top_20_atc_codes()
    mapper = NDCATCMapper.from_file("data/mappings/ndc_atc_mapping.sqlite")
    result = mapper.ndc_to_atc("38779094901")
    print(result.atc_codes)
    print(result.mappings)

if __name__ == "__main__":
    main()