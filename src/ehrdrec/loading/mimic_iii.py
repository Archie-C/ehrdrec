import hashlib
import logging
import pickle
from pathlib import Path

import pandas as pd
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
            return self._load_cache(cache_path)

        logger.info(f"Loading MIMIC-III from source: {source_path}")
        data = self._load_source(source_path)
        self._save_cache(data, cache_path)
        return data

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    
    def _cache_path(self, source_path: Path) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"mimic3_{self._cache_key(source_path)}.pkl"
    
    def _cache_key(self, source_path: Path) -> str:
        """Key based on source file mtimes — invalidates if files change."""
        mtimes = "".join(
            str((source_path / f).stat().st_mtime)
            for f in MIMIC3_FILES
            if (source_path / f).exists()
        )
        return hashlib.md5(mtimes.encode()).hexdigest()
    
    def _load_cache(self, cache_path: Path) -> LoadedData:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _save_cache(self, data: LoadedData, cache_path: Path) -> None:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cached MIMIC-III to: {cache_path}")
        
    # ------------------------------------------------------------------
    # Loading from source
    # ------------------------------------------------------------------
        
    def _load_source(self, source_path: Path) -> LoadedData:
        admissions = pd.read_csv(
            source_path / "ADMISSIONS.csv",
            usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"],
            parse_dates=["ADMITTIME", "DISCHTIME"],
        )
        diagnoses = pd.read_csv(
            source_path / "DIAGNOSES_ICD.csv",
            usecols=["HADM_ID", "ICD9_CODE"],
            dtype=str,
        )
        procedures = pd.read_csv(
            source_path / "PROCEDURES_ICD.csv",
            usecols=["HADM_ID", "ICD9_CODE"],
            dtype=str,
        )
        prescriptions = pd.read_csv(
            source_path / "PRESCRIPTIONS.csv",
            usecols=["HADM_ID", "NDC", "DRUG", "DOSE_VAL_RX", "DOSE_UNIT_RX"],
            dtype=str,
        )

        diag_grouped = self._group_codes(diagnoses)
        proc_grouped = self._group_codes(procedures)
        med_grouped = self._build_med_grouped(prescriptions)

        admissions["ADMITTIME"] = admissions["ADMITTIME"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
        admissions["DISCHTIME"] = admissions["DISCHTIME"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
        admissions["HADM_ID"] = admissions["HADM_ID"].astype(str)
        admissions["SUBJECT_ID"] = admissions["SUBJECT_ID"].astype(str)

        raw_rows = [
            {
                "patient_id": row.SUBJECT_ID,
                "admission_id": row.HADM_ID,
                "admission_time": row.ADMITTIME,
                "discharge_time": row.DISCHTIME,
                "diagnoses": diag_grouped.get(row.HADM_ID, []),
                "procedures": proc_grouped.get(row.HADM_ID, []),
                "medications": med_grouped.get(row.HADM_ID, []),
            }
            for row in admissions.itertuples()
        ]

        return LoadedData.model_validate({
            "data_source": str(source_path),
            "dataset_name": "MIMIC-III",
            "data": raw_rows,
        })

    def _group_codes(self, df: pd.DataFrame) -> dict[str, list[str]]:
        df = df.dropna(subset=["ICD9_CODE"])
        return df.groupby("HADM_ID")["ICD9_CODE"].agg(list).to_dict()

    def _build_med_grouped(self, prescriptions: pd.DataFrame) -> dict[str, list[dict]]:
        prepped = (
            prescriptions[["HADM_ID", "NDC", "DRUG", "DOSE_VAL_RX", "DOSE_UNIT_RX"]]
            .rename(columns={
                "NDC": "id",
                "DRUG": "name",
                "DOSE_VAL_RX": "dosage_value",
                "DOSE_UNIT_RX": "dosage_unit",
            })
            .fillna("")
            .apply(lambda col: col.str.strip() if col.name != "HADM_ID" else col)
        )
        return {
            str(k): v.drop(columns="HADM_ID").to_dict("records")
            for k, v in prepped.groupby("HADM_ID")
        }