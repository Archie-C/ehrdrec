import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from platformdirs import user_cache_dir

from ehrdrec.loading.base import BaseLoader
from ehrdrec.models.dataclasses.data_loading import LoadedData

logger = logging.getLogger(__name__)

# MIMIC-IV's hosp module ships these gzipped by default; we also accept the
# decompressed .csv form (whichever exists on disk wins).
MIMIC4_FILES = ["admissions.csv", "diagnoses_icd.csv", "procedures_icd.csv", "prescriptions.csv"]


class MIMIC4Loader(BaseLoader):
    """Loads the MIMIC-IV 'hosp' module (admissions, diagnoses_icd,
    procedures_icd, prescriptions) into the shared LoadedData schema.

    `source` should point at the directory containing those four files,
    e.g. ``mimic-iv-3.1/hosp``. Files may be plain ``.csv`` or ``.csv.gz``.

    Unlike MIMIC-III, MIMIC-IV mixes ICD-9 and ICD-10 diagnosis/procedure
    codes (see the ``icd_version`` column). The same code string can mean
    different things under the two systems, so codes are stored as
    ``"{icd_version}:{icd_code}"`` (e.g. ``"9:4019"``, ``"10:I10"``) to keep
    them from colliding in the downstream vocab.
    """

    def __init__(self, cache_dir: Path | None = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(user_cache_dir("ehrdrec"))

    def load(self, source: str, force_reload: bool = False) -> LoadedData:
        source_path = Path(source)
        cache_path = self._cache_path(source_path)

        if not force_reload and cache_path.exists():
            logger.info(f"Loading MIMIC-IV from cache: {cache_path}")
            return LoadedData(
                data_source=str(source_path),
                dataset_name="MIMIC-IV",
                frame=pl.scan_parquet(cache_path),
            )

        logger.info(f"Loading MIMIC-IV from source: {source_path}")
        frame = self._load_source(source_path)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame.collect().write_parquet(cache_path, compression="zstd", compression_level=3)
        logger.info(f"Cached MIMIC-IV to: {cache_path}")

        return LoadedData(
            data_source=str(source_path),
            dataset_name="MIMIC-IV",
            frame=pl.scan_parquet(cache_path),
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, source_path: Path) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"mimic4_{self._cache_key(source_path)}.parquet"

    def _cache_key(self, source_path: Path) -> str:
        """Key based on source file mtimes — invalidates if files change.

        Raises FileNotFoundError if any expected file is missing, so the
        cache key can never silently omit a file.
        """
        mtimes = "".join(
            str(self._resolve(source_path, f).stat().st_mtime)
            for f in MIMIC4_FILES
        )
        return hashlib.md5(mtimes.encode()).hexdigest()

    @staticmethod
    def _resolve(source_path: Path, filename: str) -> Path:
        """Returns the .csv path if it exists, else the .csv.gz path.

        Raises FileNotFoundError if neither is present.
        """
        plain = source_path / filename
        if plain.exists():
            return plain
        gz = source_path / f"{filename}.gz"
        if gz.exists():
            return gz
        raise FileNotFoundError(f"Could not find {filename} or {filename}.gz in {source_path}")

    # ------------------------------------------------------------------
    # Loading from source
    # ------------------------------------------------------------------

    def _load_source(self, source_path: Path) -> pl.LazyFrame:
        # Read all four files in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_admissions    = pool.submit(self._read_admissions,    source_path)
            f_diagnoses     = pool.submit(self._read_codes,         source_path, "diagnoses_icd.csv")
            f_procedures    = pool.submit(self._read_codes,         source_path, "procedures_icd.csv")
            f_prescriptions = pool.submit(self._read_prescriptions, source_path)

            admissions    = f_admissions.result()
            diagnoses     = f_diagnoses.result()
            procedures    = f_procedures.result()
            prescriptions = f_prescriptions.result()

        # Group diagnoses and procedures into List[Utf8] per admission
        diag_grouped = (
            diagnoses
            .group_by("hadm_id")
            .agg(pl.col("icd_code").alias("diagnoses"))
        )
        proc_grouped = (
            procedures
            .group_by("hadm_id")
            .agg(pl.col("icd_code").alias("procedures"))
        )

        # Group prescriptions into List[Struct] per admission
        med_grouped = (
            prescriptions
            .group_by("hadm_id")
            .agg(
                pl.struct(
                    pl.col("NDC"),
                    pl.col("name"),
                    pl.col("dosage_value"),
                    pl.col("dosage_unit"),
                ).alias("medications")
            )
        )

        # Join everything onto admissions.
        # Inner join on med_grouped: only keep admissions with >=1 medication record.
        result = (
            admissions
            .join(med_grouped,  on="hadm_id", how="inner")
            .join(diag_grouped, on="hadm_id", how="left")
            .join(proc_grouped, on="hadm_id", how="left")
            # Admissions with no diagnoses / procedures get empty lists
            .with_columns([
                pl.col("diagnoses").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
                pl.col("procedures").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
            ])
            .rename({
                "subject_id": "patient_id",
                "hadm_id":    "admission_id",
                "admittime":  "admission_time",
                "dischtime":  "discharge_time",
            })
            .select([
                "patient_id",
                "admission_id",
                "admission_time",
                "discharge_time",
                "diagnoses",
                "procedures",
                "medications",
            ])
        )

        return result.lazy()

    # ------------------------------------------------------------------
    # Per-file readers
    # ------------------------------------------------------------------

    def _read_admissions(self, source_path: Path) -> pl.DataFrame:
        return (
            pl.read_csv(
                self._resolve(source_path, "admissions.csv"),
                columns=["subject_id", "hadm_id", "admittime", "dischtime"],
                schema_overrides={
                    "subject_id": pl.Utf8,
                    "hadm_id":    pl.Utf8,
                    "admittime":  pl.Utf8,
                    "dischtime":  pl.Utf8,
                },
                null_values=[""],
            )
            .with_columns([
                pl.col("admittime")
                  .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                  .dt.strftime("%Y-%m-%dT%H:%M:%S")
                  .fill_null(""),
                pl.col("dischtime")
                  .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                  .dt.strftime("%Y-%m-%dT%H:%M:%S")
                  .fill_null(""),
            ])
        )

    def _read_codes(self, source_path: Path, filename: str) -> pl.DataFrame:
        return (
            pl.read_csv(
                self._resolve(source_path, filename),
                columns=["hadm_id", "icd_code", "icd_version"],
                schema_overrides={"hadm_id": pl.Utf8, "icd_code": pl.Utf8, "icd_version": pl.Utf8},
                null_values=[""],
            )
            .drop_nulls("icd_code")
            # Disambiguate ICD-9 vs ICD-10 codes that share a string but
            # mean different things (e.g. ICD-9 "4019" vs unrelated ICD-10 codes).
            .with_columns(
                (pl.col("icd_version") + pl.lit(":") + pl.col("icd_code")).alias("icd_code")
            )
            .select(["hadm_id", "icd_code"])
        )

    def _read_prescriptions(self, source_path: Path) -> pl.DataFrame:
        return (
            pl.read_csv(
                self._resolve(source_path, "prescriptions.csv"),
                columns=["hadm_id", "ndc", "drug", "dose_val_rx", "dose_unit_rx"],
                schema_overrides={
                    "hadm_id":      pl.Utf8,
                    "ndc":          pl.Utf8,
                    "drug":         pl.Utf8,
                    "dose_val_rx":  pl.Utf8,
                    "dose_unit_rx": pl.Utf8,
                },
                null_values=[""],
            )
            .rename({
                "ndc":          "NDC",
                "drug":         "name",
                "dose_val_rx":  "dosage_value",
                "dose_unit_rx": "dosage_unit",
            })
            .with_columns([
                pl.col("NDC").str.strip_chars(),
                pl.col("name").str.strip_chars().fill_null(""),
                pl.col("dosage_value").str.strip_chars().fill_null(""),
                pl.col("dosage_unit").str.strip_chars().fill_null(""),
            ])
        )
