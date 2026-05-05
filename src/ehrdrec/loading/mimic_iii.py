import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
 
import polars as pl
from platformdirs import user_cache_dir
 
from ehrdrec.loading.base import BaseLoader
from ehrdrec.models.data_loading import LoadedData
 
logger = logging.getLogger(__name__)

MIMIC3_FILES = ["ADMISSIONS.csv", "DIAGNOSES_ICD.csv", "PROCEDURES_ICD.csv", "PRESCRIPTIONS.csv"]

class MIMIC3Loader(BaseLoader):
    def __init__(self, cache_dir: Path | None = None):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(user_cache_dir("ehrdrec"))

    def load(self, source: str, force_reload: bool = False) -> LoadedData:
        source_path = Path(source)
        cache_path = self._cache_path(source_path)

        if not force_reload and cache_path.exists():
            logger.info(f"Loading MIMIC-III from cache: {cache_path}")
            return LoadedData(
                data_source=str(source_path),
                dataset_name="MIMIC-III",
                frame=pl.scan_parquet(cache_path),
            )

        logger.info(f"Loading MIMIC-III from source: {source_path}")
        frame = self._load_source(source_path)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Collect once to write; caller always gets a LazyFrame over parquet
        frame.collect().write_parquet(cache_path, compression="zstd", compression_level=3)
        logger.info(f"Cached MIMIC-III to: {cache_path}")
 
        return LoadedData(
            data_source=str(source_path),
            dataset_name="MIMIC-III",
            frame=pl.scan_parquet(cache_path),
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, source_path: Path) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"mimic3_{self._cache_key(source_path)}.parquet"

    def _cache_key(self, source_path: Path) -> str:
        """Key based on source file mtimes — invalidates if files change.

        Raises FileNotFoundError if any expected file is missing, so the
        cache key can never silently omit a file.
        """
        mtimes = "".join(
            str((source_path / f).stat().st_mtime) 
            for f in MIMIC3_FILES
        )
        return hashlib.md5(mtimes.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Loading from source
    # ------------------------------------------------------------------

    def _load_source(self, source_path: Path) -> pl.LazyFrame:
        # Read all four CSVs in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_admissions    = pool.submit(self._read_admissions,    source_path)
            f_diagnoses     = pool.submit(self._read_codes,         source_path / "DIAGNOSES_ICD.csv")
            f_procedures    = pool.submit(self._read_codes,         source_path / "PROCEDURES_ICD.csv")
            f_prescriptions = pool.submit(self._read_prescriptions, source_path)
 
            admissions    = f_admissions.result()
            diagnoses     = f_diagnoses.result()
            procedures    = f_procedures.result()
            prescriptions = f_prescriptions.result()
 
        # Group diagnoses and procedures into List[Utf8] per admission
        diag_grouped = (
            diagnoses
            .group_by("HADM_ID")
            .agg(pl.col("ICD9_CODE").alias("diagnoses"))
        )
        proc_grouped = (
            procedures
            .group_by("HADM_ID")
            .agg(pl.col("ICD9_CODE").alias("procedures"))
        )
 
        # Group prescriptions into List[Struct] per admission
        med_grouped = (
            prescriptions
            .group_by("HADM_ID")
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
        # Inner join on med_grouped
        # only keep admissions that have at least one medication record.
        result = (
            admissions
            .join(med_grouped,  on="HADM_ID", how="inner")
            .join(diag_grouped, on="HADM_ID", how="left")
            .join(proc_grouped, on="HADM_ID", how="left")
            # Admissions with no diagnoses / procedures get empty lists
            .with_columns([
                pl.col("diagnoses").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
                pl.col("procedures").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))),
            ])
            .rename({
                "SUBJECT_ID": "patient_id",
                "HADM_ID":    "admission_id",
                "ADMITTIME":  "admission_time",
                "DISCHTIME":  "discharge_time",
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
 
    @staticmethod
    def _read_admissions(source_path: Path) -> pl.DataFrame:
        return (
            pl.read_csv(
                source_path / "ADMISSIONS.csv",
                columns=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"],
                schema_overrides={
                    "SUBJECT_ID": pl.Utf8,
                    "HADM_ID":    pl.Utf8,
                    "ADMITTIME":  pl.Utf8,
                    "DISCHTIME":  pl.Utf8,
                },
                null_values=[""],
            )
            .with_columns([
                pl.col("ADMITTIME")
                  .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                  .dt.strftime("%Y-%m-%dT%H:%M:%S")
                  .fill_null(""),
                pl.col("DISCHTIME")
                  .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                  .dt.strftime("%Y-%m-%dT%H:%M:%S")
                  .fill_null(""),
            ])
        )
 
    @staticmethod
    def _read_codes(path: Path) -> pl.DataFrame:
        return (
            pl.read_csv(
                path,
                columns=["HADM_ID", "ICD9_CODE"],
                schema_overrides={"HADM_ID": pl.Utf8, "ICD9_CODE": pl.Utf8},
                null_values=[""],
            )
            .drop_nulls("ICD9_CODE")
        )
    
    @staticmethod
    def _read_prescriptions(source_path: Path) -> pl.DataFrame:
        return (
            pl.read_csv(
                source_path / "PRESCRIPTIONS.csv",
                columns=["HADM_ID", "NDC", "DRUG", "DOSE_VAL_RX", "DOSE_UNIT_RX"],
                schema_overrides={
                    "HADM_ID":      pl.Utf8,
                    "NDC":          pl.Utf8,
                    "DRUG":         pl.Utf8,
                    "DOSE_VAL_RX":  pl.Utf8,
                    "DOSE_UNIT_RX": pl.Utf8,
                },
                null_values=[""],
            )
            .rename({
                "DRUG":         "name",
                "DOSE_VAL_RX":  "dosage_value",
                "DOSE_UNIT_RX": "dosage_unit",
            })
            .with_columns([
                pl.col("NDC").str.strip_chars(),
                pl.col("name").str.strip_chars().fill_null(""),
                pl.col("dosage_value").str.strip_chars().fill_null(""),
                pl.col("dosage_unit").str.strip_chars().fill_null(""),
            ])
        )