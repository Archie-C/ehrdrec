from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import polars as pl

from ehrdrec.requirements import (
    DataRequest,
    DataRequirement,
)


class MIMIC3Files(Enum):
    ADMISSIONS = "ADMISSIONS.csv"
    CALLOUT = "CALLOUT.csv"
    CAREGIVERS = "CAREGIVERS.csv"
    CHARTEVENTS = "CHARTEVENTS.csv"
    CPTEVENTS = "CPTEVENTS.csv"
    D_CPT = "D_CPT.csv"
    D_ICD_DIAGNOSES = "D_ICD_DIAGNOSES.csv"
    D_ICD_PROCEDURES = "D_ICD_PROCEDURES.csv"
    D_ITEMS = "D_ITEMS.csv"
    D_LABITEMS = "D_LABITEMS.csv"
    DATETIMEEVENTS = "DATETIMEEVENTS.csv"
    DIAGNOSES_ICD = "DIAGNOSES_ICD.csv"
    DRGCODES = "DRGCODES.csv"
    ICUSTAYS = "ICUSTAYS.csv"
    INPUTEVENTS_CV = "INPUTEVENTS_CV.csv"
    INPUTEVENTS_MV = "INPUTEVENTS_MV.csv"
    LABEVENTS = "LABEVENTS.csv"
    MICROBIOLOGYEVENTS = "MICROBIOLOGYEVENTS.csv"
    NOTEEVENTS = "NOTEEVENTS.csv"
    OUTPUTEVENTS = "OUTPUTEVENTS.csv"
    PATIENTS = "PATIENTS.csv"
    PRESCRIPTIONS = "PRESCRIPTIONS.csv"
    PROCEDUREEVENTS_MV = "PROCEDUREEVENTS_MV.csv"
    PROCEDURES_ICD = "PROCEDURES_ICD.csv"
    SERVICES = "SERVICES.csv"
    TRANSFERS = "TRANSFERS.csv"


@dataclass(frozen=True)
class FrameRequest:
    file: MIMIC3Files
    columns: frozenset[str]


class MIMIC3Loader:
    """
    Provides lazy access to MIMIC-III.

    This class knows:
        DataRequirement -> MIMIC-III file + columns

    It does NOT know:
        TaskRequirement
        Feature
        Representation
        InputStructure
        ModelRequirement
    """

    _requirement_map: dict[
        DataRequirement,
        tuple[FrameRequest, ...],
    ] = {

        DataRequirement.VISIT_TIMES: (
            FrameRequest(
                file=MIMIC3Files.ADMISSIONS,
                columns=frozenset({
                    "SUBJECT_ID",
                    "HADM_ID",
                    "ADMITTIME",
                }),
            ),
        ),

        DataRequirement.DIAGNOSES: (
            FrameRequest(
                file=MIMIC3Files.DIAGNOSES_ICD,
                columns=frozenset({
                    "ROW_ID",
                    "SUBJECT_ID",
                    "HADM_ID",
                    "SEQ_NUM",
                    "ICD9_CODE",
                }),
            ),
        ),

        DataRequirement.PROCEDURES: (
            FrameRequest(
                file=MIMIC3Files.PROCEDURES_ICD,
                columns=frozenset({
                    "ROW_ID",
                    "SUBJECT_ID",
                    "HADM_ID",
                    "SEQ_NUM",
                    "ICD9_CODE",
                }),
            ),
        ),

        DataRequirement.MEDICATIONS: (
            FrameRequest(
                file=MIMIC3Files.PRESCRIPTIONS,
                columns=frozenset({
                    "ROW_ID",
                    "SUBJECT_ID",
                    "HADM_ID",
                    "ICUSTAY_ID",
                    "STARTDATE",
                    "ENDDATE",
                    "DRUG_TYPE",
                    "DRUG",
                    "DRUG_NAME_POE",
                    "DRUG_NAME_GENERIC",
                    "FORMULARY_DRUG_CD",
                    "GSN",
                    "NDC",
                    "PROD_STRENGTH",
                    "DOSE_VAL_RX",
                    "DOSE_UNIT_RX",
                    "FORM_VAL_DISP",
                    "FORM_UNIT_DISP",
                    "ROUTE",
                }),
            ),
        ),
    }

    _schema_overrides: dict[
        MIMIC3Files,
        dict[str, pl.DataType],
    ] = {
        # Codes should remain codes rather than accidentally being
        # inferred as numeric values.
        MIMIC3Files.DIAGNOSES_ICD: {
            "ICD9_CODE": pl.String,
        },

        MIMIC3Files.PROCEDURES_ICD: {
            "ICD9_CODE": pl.String,
        },

        MIMIC3Files.PRESCRIPTIONS: {
            "NDC": pl.String,
            "GSN": pl.String,
            "FORMULARY_DRUG_CD": pl.String,
        },
    }

    def load(
        self,
        path: str | Path,
        request: DataRequest,
    ) -> dict[str, pl.LazyFrame]:

        path = Path(path)

        resolved = self._resolve_request(request)

        self._validate(
            path=path,
            required_files=set(resolved),
        )

        return {
            file.name: self._load_frame(
                path=path,
                file=file,
                columns=columns,
            )
            for file, columns in resolved.items()
        }

    def _resolve_request(
        self,
        request: DataRequest,
    ) -> dict[MIMIC3Files, set[str]]:

        resolved: dict[
            MIMIC3Files,
            set[str],
        ] = {}

        for requirement in request.requirements:

            try:
                frame_requests = self._requirement_map[
                    requirement
                ]
            except KeyError:
                raise ValueError(
                    "MIMIC-III does not support "
                    f"{requirement.name}"
                )

            for frame_request in frame_requests:
                resolved.setdefault(
                    frame_request.file,
                    set(),
                ).update(
                    frame_request.columns
                )

        return resolved

    def _load_frame(
        self,
        path: Path,
        file: MIMIC3Files,
        columns: set[str],
    ) -> pl.LazyFrame:

        return (
            pl.scan_csv(
                path / file.value,
                infer_schema_length=1000,
                low_memory=True,
                try_parse_dates=True,
                schema_overrides=self._schema_overrides.get(
                    file
                ),
            )
            .select(sorted(columns))
        )

    def _validate(
        self,
        path: Path,
        required_files: set[MIMIC3Files],
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {path}"
            )

        if not path.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {path}"
            )

        missing = [
            file.value
            for file in required_files
            if not (path / file.value).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required MIMIC-III files: "
                + ", ".join(sorted(missing))
            )