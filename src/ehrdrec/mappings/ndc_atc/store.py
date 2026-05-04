import sqlite3
from pathlib import Path

from .models import ATCMapping, MappingResult
from .normalise import atc_to_level


class SQLiteMappingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)

    def get_metadata(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM metadata"
        ).fetchall()
        return dict(rows)

    def lookup_ndc(
        self,
        normalised_ndc: str,
        input_ndc: str,
        atc_level: int | None = None,
    ) -> MappingResult:
        rows = self.conn.execute(
            """
            SELECT DISTINCT
                drug_rxcui,
                ingredient_rxcui,
                atc_code,
                atc_name,
                match_type
            FROM ndc_to_atc
            WHERE ndc = ?
            ORDER BY drug_rxcui, ingredient_rxcui, atc_code, match_type
            """,
            (normalised_ndc,),
        ).fetchall()

        metadata = self.get_metadata()

        grouped = {}

        for row in rows:
            drug_rxcui, ingredient_rxcui, atc_code, atc_name, match_type = row

            if atc_level is not None:
                atc_code = atc_to_level(atc_code, atc_level)

            key = (drug_rxcui, ingredient_rxcui, atc_code)

            if key not in grouped:
                grouped[key] = {
                    "drug_rxcui": drug_rxcui,
                    "ingredient_rxcui": ingredient_rxcui,
                    "atc_code": atc_code,
                    "atc_name": atc_name,
                    "match_types": set(),
                }

            grouped[key]["match_types"].add(match_type)

        mappings = [
            ATCMapping(
                drug_rxcui=v["drug_rxcui"],
                ingredient_rxcui=v["ingredient_rxcui"],
                atc_code=v["atc_code"],
                atc_name=v["atc_name"],
                match_types=tuple(sorted(v["match_types"])),
            )
            for v in grouped.values()
        ]

        return MappingResult(
            input_ndc=input_ndc,
            normalised_ndc=normalised_ndc,
            mappings=mappings,
            mapping_version=metadata["mapping_version"],
        )

    def close(self) -> None:
        self.conn.close()