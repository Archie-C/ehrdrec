from pathlib import Path
import sqlite3
import csv
from datetime import datetime, timezone

from .normalise import normalise_ndc


class MappingBuilder:
    def __init__(
        self,
        umls_dir: str | Path,
        output_path: str | Path,
        mapping_version: str,
    ):
        self.umls_dir = Path(umls_dir)
        self.output_path = Path(output_path)
        self.mapping_version = mapping_version

    def build(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.output_path)

        self._create_schema(conn)
        self._insert_metadata(conn)

        self._load_ndc_to_rxcui(conn)
        self._load_rxcui_to_ingredient_direct(conn)
        self._load_rxcui_to_group(conn)
        self._load_pack_to_drug(conn)
        self._load_rxcui_to_atc(conn)

        self._resolve_direct_ingredient(conn)
        self._resolve_group_ingredient(conn)
        self._resolve_pack_ingredient(conn)

        conn.commit()
        conn.close()

    def _rrf_path(self, filename: str) -> Path:
        path = self.umls_dir / "rrf" / filename
        if not path.exists():
            path = self.umls_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Could not find {filename} in {self.umls_dir}")
        return path

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS ndc_to_rxcui;
            DROP TABLE IF EXISTS rxcui_to_ingredient;
            DROP TABLE IF EXISTS rxcui_to_group;
            DROP TABLE IF EXISTS pack_to_drug;
            DROP TABLE IF EXISTS rxcui_to_atc;
            DROP TABLE IF EXISTS ndc_to_atc;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE ndc_to_rxcui (
                ndc TEXT NOT NULL,
                raw_ndc TEXT NOT NULL,
                rxcui TEXT NOT NULL,
                source TEXT,
                suppress TEXT,
                PRIMARY KEY (ndc, raw_ndc, rxcui)
            );

            CREATE TABLE rxcui_to_ingredient (
                drug_rxcui TEXT NOT NULL,
                ingredient_rxcui TEXT NOT NULL,
                relationship TEXT NOT NULL,
                source TEXT,
                PRIMARY KEY (drug_rxcui, ingredient_rxcui, relationship)
            );

            CREATE TABLE rxcui_to_group (
                drug_rxcui TEXT NOT NULL,
                group_rxcui TEXT NOT NULL,
                relationship TEXT NOT NULL,
                source TEXT,
                PRIMARY KEY (drug_rxcui, group_rxcui, relationship)
            );

            CREATE TABLE pack_to_drug (
                pack_rxcui TEXT NOT NULL,
                drug_rxcui TEXT NOT NULL,
                relationship TEXT NOT NULL,
                source TEXT,
                PRIMARY KEY (pack_rxcui, drug_rxcui, relationship)
            );

            CREATE TABLE rxcui_to_atc (
                ingredient_rxcui TEXT NOT NULL,
                atc_code TEXT NOT NULL,
                atc_name TEXT,
                tty TEXT,
                suppress TEXT,
                PRIMARY KEY (ingredient_rxcui, atc_code)
            );

            CREATE TABLE ndc_to_atc (
                ndc TEXT NOT NULL,
                raw_ndc TEXT NOT NULL,
                drug_rxcui TEXT NOT NULL,
                ingredient_rxcui TEXT NOT NULL,
                atc_code TEXT NOT NULL,
                atc_name TEXT,
                match_type TEXT NOT NULL,
                PRIMARY KEY (
                    ndc,
                    raw_ndc,
                    drug_rxcui,
                    ingredient_rxcui,
                    atc_code,
                    match_type
                )
            );
            """
        )

    def _insert_metadata(self, conn: sqlite3.Connection) -> None:
        rows = [
            ("mapping_version", self.mapping_version),
            ("created_at", datetime.now(timezone.utc).isoformat()),
            ("builder", "MappingBuilder"),
        ]
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", rows)

    def _load_ndc_to_rxcui(self, conn: sqlite3.Connection) -> None:
        path = self._rrf_path("RXNSAT.RRF")
        rows_to_insert = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 13:
                    continue

                rxcui = row[0]
                atn = row[8]
                sab = row[9]
                atv = row[10]
                suppress = row[11]

                if atn != "NDC":
                    continue

                if suppress != "N":
                    continue

                raw_ndc = atv.strip()

                try:
                    ndc = normalise_ndc(raw_ndc)
                except Exception:
                    continue

                rows_to_insert.append((ndc, raw_ndc, rxcui, sab, suppress))

        conn.executemany(
            """
            INSERT OR IGNORE INTO ndc_to_rxcui (
                ndc,
                raw_ndc,
                rxcui,
                source,
                suppress
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_rxcui_ndc "
            "ON ndc_to_rxcui(ndc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_rxcui_rxcui "
            "ON ndc_to_rxcui(rxcui)"
        )

    def _load_rxcui_to_ingredient_direct(self, conn: sqlite3.Connection) -> None:
        path = self._rrf_path("RXNREL.RRF")
        rows_to_insert = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 16:
                    continue

                rxcui1 = row[0]
                rxcui2 = row[4]
                rela = row[7]
                sab = row[10]
                suppress = row[14]

                if suppress == "O":
                    continue

                # RELA = relationship RXCUI2 has to RXCUI1
                if rela == "has_ingredient":
                    drug_rxcui = rxcui2
                    ingredient_rxcui = rxcui1
                elif rela == "ingredient_of":
                    ingredient_rxcui = rxcui2
                    drug_rxcui = rxcui1
                else:
                    continue

                rows_to_insert.append(
                    (drug_rxcui, ingredient_rxcui, rela, sab)
                )

        conn.executemany(
            """
            INSERT OR IGNORE INTO rxcui_to_ingredient (
                drug_rxcui,
                ingredient_rxcui,
                relationship,
                source
            )
            VALUES (?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    def _load_rxcui_to_group(self, conn: sqlite3.Connection) -> None:
        path = self._rrf_path("RXNREL.RRF")
        rows_to_insert = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 16:
                    continue

                rxcui1 = row[0]
                rxcui2 = row[4]
                rela = row[7]
                sab = row[10]
                suppress = row[14]

                if suppress == "O":
                    continue

                # RELA = relationship RXCUI2 has to RXCUI1
                if rela == "isa":
                    child_rxcui = rxcui2
                    parent_rxcui = rxcui1
                elif rela == "inverse_isa":
                    parent_rxcui = rxcui2
                    child_rxcui = rxcui1
                else:
                    continue

                rows_to_insert.append(
                    (child_rxcui, parent_rxcui, rela, sab)
                )

        conn.executemany(
            """
            INSERT OR IGNORE INTO rxcui_to_group (
                drug_rxcui,
                group_rxcui,
                relationship,
                source
            )
            VALUES (?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    def _load_pack_to_drug(self, conn: sqlite3.Connection) -> None:
        path = self._rrf_path("RXNREL.RRF")
        rows_to_insert = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 16:
                    continue

                rxcui1 = row[0]
                rxcui2 = row[4]
                rela = row[7]
                sab = row[10]
                suppress = row[14]

                if suppress == "O":
                    continue

                # RELA = relationship RXCUI2 has to RXCUI1
                if rela == "contains":
                    pack_rxcui = rxcui2
                    drug_rxcui = rxcui1
                elif rela == "contained_in":
                    drug_rxcui = rxcui2
                    pack_rxcui = rxcui1
                else:
                    continue

                rows_to_insert.append(
                    (pack_rxcui, drug_rxcui, rela, sab)
                )

        conn.executemany(
            """
            INSERT OR IGNORE INTO pack_to_drug (
                pack_rxcui,
                drug_rxcui,
                relationship,
                source
            )
            VALUES (?, ?, ?, ?)
            """,
            rows_to_insert,
        )

    def _load_rxcui_to_atc(self, conn: sqlite3.Connection) -> None:
        path = self._rrf_path("RXNCONSO.RRF")
        rows_to_insert = []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 17:
                    continue

                rxcui = row[0]
                sab = row[11]
                tty = row[12]
                code = row[13]
                name = row[14]
                suppress = row[16]

                if sab != "ATC":
                    continue

                if suppress == "O":
                    continue

                rows_to_insert.append(
                    (rxcui, code, name, tty, suppress)
                )

        conn.executemany(
            """
            INSERT OR IGNORE INTO rxcui_to_atc (
                ingredient_rxcui,
                atc_code,
                atc_name,
                tty,
                suppress
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rxcui_to_atc_ingredient "
            "ON rxcui_to_atc(ingredient_rxcui)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rxcui_to_atc_code "
            "ON rxcui_to_atc(atc_code)"
        )

    def _resolve_direct_ingredient(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO ndc_to_atc (
                ndc,
                raw_ndc,
                drug_rxcui,
                ingredient_rxcui,
                atc_code,
                atc_name,
                match_type
            )
            SELECT
                n.ndc,
                n.raw_ndc,
                n.rxcui,
                i.ingredient_rxcui,
                a.atc_code,
                a.atc_name,
                'direct_ingredient'
            FROM ndc_to_rxcui n
            JOIN rxcui_to_ingredient i
              ON n.rxcui = i.drug_rxcui
            JOIN rxcui_to_atc a
              ON i.ingredient_rxcui = a.ingredient_rxcui
            """
        )

    def _resolve_group_ingredient(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO ndc_to_atc (
                ndc,
                raw_ndc,
                drug_rxcui,
                ingredient_rxcui,
                atc_code,
                atc_name,
                match_type
            )
            SELECT
                n.ndc,
                n.raw_ndc,
                n.rxcui,
                i.ingredient_rxcui,
                a.atc_code,
                a.atc_name,
                'group_ingredient'
            FROM ndc_to_rxcui n
            JOIN rxcui_to_group g
              ON n.rxcui = g.drug_rxcui
            JOIN rxcui_to_ingredient i
              ON g.group_rxcui = i.drug_rxcui
            JOIN rxcui_to_atc a
              ON i.ingredient_rxcui = a.ingredient_rxcui
            """
        )

    def _resolve_pack_ingredient(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO ndc_to_atc (
                ndc,
                raw_ndc,
                drug_rxcui,
                ingredient_rxcui,
                atc_code,
                atc_name,
                match_type
            )
            SELECT
                n.ndc,
                n.raw_ndc,
                n.rxcui,
                i.ingredient_rxcui,
                a.atc_code,
                a.atc_name,
                'pack_ingredient'
            FROM ndc_to_rxcui n
            JOIN pack_to_drug p
              ON n.rxcui = p.pack_rxcui
            JOIN rxcui_to_ingredient i
              ON p.drug_rxcui = i.drug_rxcui
            JOIN rxcui_to_atc a
              ON i.ingredient_rxcui = a.ingredient_rxcui
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_atc_ndc "
            "ON ndc_to_atc(ndc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_atc_drug "
            "ON ndc_to_atc(drug_rxcui)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_atc_ingredient "
            "ON ndc_to_atc(ingredient_rxcui)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ndc_to_atc_atc "
            "ON ndc_to_atc(atc_code)"
        )