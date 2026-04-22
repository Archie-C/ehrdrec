import pandas as pd
from pathlib import Path

from ehrdrec.loading.base import BaseLoader
from ehrdrec.models.base import ExtendedMedication
from ehrdrec.models.data_loading import LoadedData, LoadedDataRow

class MIMIC3Loader(BaseLoader):
    def load(self, source: str):
        source_path = Path(source)
        admissions = pd.read_csv(source_path / "ADMISSIONS.csv", parse_dates=["ADMITTIME", "DISCHTIME"])
        diagnoses  = pd.read_csv(source_path / "DIAGNOSES_ICD.csv", dtype=str)
        procedures = pd.read_csv(source_path / "PROCEDURES_ICD.csv", dtype=str)
        prescriptions = pd.read_csv(source_path / "PRESCRIPTIONS.csv", dtype=str)
        
        diag_grouped = (
            diagnoses.groupby("HADM_ID")["ICD9_CODE"]
            .apply(lambda codes: [c for c in codes if pd.notna(c)])
            .to_dict()
        )

        proc_grouped = (
            procedures.groupby("HADM_ID")["ICD9_CODE"]
            .apply(lambda codes: [c for c in codes if pd.notna(c)])
            .to_dict()
        )
        med_grouped: dict[str, list[ExtendedMedication]] = {
            str(hadm_id): self._build_medications(group)
            for hadm_id, group in prescriptions.groupby("HADM_ID")
        }
        
        rows = self._build_rows(admissions, diag_grouped, proc_grouped, med_grouped)
        return LoadedData(
            data_source=source,
            dataset_name="MIMIC-III",
            data=rows
        )
    
    def _build_medications(self, prescriptions: pd.DataFrame) -> list[ExtendedMedication]:
        meds = []
        for _, rx in prescriptions.iterrows():
            drug_id = rx.get("NDC")
            meds.append(ExtendedMedication(
                id=str(drug_id).strip(),
                name=str(rx.get("DRUG", "")).strip(),
                dosage_value=str(rx.get("DOSE_VAL_RX", "")).strip(),
                dosage_unit=str(rx.get("DOSE_UNIT_RX", "")).strip(),
            ))
        return meds
    
    def _build_rows(
        self, 
        admissions: pd.DataFrame, 
        diag_grouped: dict[str, list[str]], 
        proc_grouped: dict[str, list[str]], 
        med_grouped: dict[str, list[ExtendedMedication]]
    ) -> list[LoadedDataRow]:
        rows: list[LoadedDataRow] = []
        for _, adm in admissions.iterrows():
            hadm_id = str(adm["HADM_ID"])
            rows.append(LoadedDataRow(
                patient_id=str(adm["SUBJECT_ID"]),
                admission_id=hadm_id,
                admission_time=adm["ADMITTIME"].isoformat() if pd.notna(adm["ADMITTIME"]) else "",
            discharge_time=adm["DISCHTIME"].isoformat() if pd.notna(adm["DISCHTIME"]) else "",
            diagnoses=diag_grouped.get(hadm_id, []),
            procedures=proc_grouped.get(hadm_id, []),
            medications=med_grouped.get(hadm_id, []),
        ))
        return rows

        