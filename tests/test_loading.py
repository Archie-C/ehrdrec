from pathlib import Path
from textwrap import dedent

from ehrdrec.loading.mimic_iii import MIMIC3Loader
from ehrdrec.loading.mimic_iv import MIMIC4Loader


def _write(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def test_mimic3_loader_adds_icd_prompt_terms(tmp_path):
    _write(
        tmp_path / "ADMISSIONS.csv",
        """
        SUBJECT_ID,HADM_ID,ADMITTIME,DISCHTIME
        1,10,2020-01-01 00:00:00,2020-01-02 00:00:00
        """,
    )
    _write(
        tmp_path / "DIAGNOSES_ICD.csv",
        """
        HADM_ID,ICD9_CODE
        10,4019
        """,
    )
    _write(
        tmp_path / "PROCEDURES_ICD.csv",
        """
        HADM_ID,ICD9_CODE
        10,3995
        """,
    )
    _write(
        tmp_path / "D_ICD_DIAGNOSES.csv",
        """
        ICD9_CODE,LONG_TITLE
        4019,Unspecified essential hypertension
        """,
    )
    _write(
        tmp_path / "D_ICD_PROCEDURES.csv",
        """
        ICD9_CODE,LONG_TITLE
        3995,Hemodialysis
        """,
    )
    _write(
        tmp_path / "PRESCRIPTIONS.csv",
        """
        HADM_ID,NDC,DRUG,DOSE_VAL_RX,DOSE_UNIT_RX
        10,123,Metformin,1,TAB
        """,
    )

    row = MIMIC3Loader()._load_source(tmp_path).collect().row(0, named=True)

    assert row["diagnoses"] == ["4019"]
    assert row["procedures"] == ["3995"]
    assert row["diagnosis_terms"] == ["4019 - Unspecified essential hypertension"]
    assert row["procedure_terms"] == ["3995 - Hemodialysis"]


def test_mimic4_loader_adds_icd_prompt_terms(tmp_path):
    _write(
        tmp_path / "admissions.csv",
        """
        subject_id,hadm_id,admittime,dischtime
        1,10,2020-01-01 00:00:00,2020-01-02 00:00:00
        """,
    )
    _write(
        tmp_path / "diagnoses_icd.csv",
        """
        hadm_id,icd_code,icd_version
        10,I10,10
        """,
    )
    _write(
        tmp_path / "procedures_icd.csv",
        """
        hadm_id,icd_code,icd_version
        10,5A1D70Z,10
        """,
    )
    _write(
        tmp_path / "d_icd_diagnoses.csv",
        """
        icd_code,icd_version,long_title
        I10,10,Essential hypertension
        """,
    )
    _write(
        tmp_path / "d_icd_procedures.csv",
        """
        icd_code,icd_version,long_title
        5A1D70Z,10,Performance of urinary filtration
        """,
    )
    _write(
        tmp_path / "prescriptions.csv",
        """
        hadm_id,ndc,drug,dose_val_rx,dose_unit_rx
        10,123,Metformin,1,TAB
        """,
    )

    row = MIMIC4Loader()._load_source(tmp_path).collect().row(0, named=True)

    assert row["diagnoses"] == ["10:I10"]
    assert row["procedures"] == ["10:5A1D70Z"]
    assert row["diagnosis_terms"] == ["ICD-10 I10 - Essential hypertension"]
    assert row["procedure_terms"] == ["ICD-10 5A1D70Z - Performance of urinary filtration"]
