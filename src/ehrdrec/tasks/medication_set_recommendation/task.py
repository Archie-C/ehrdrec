from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

import polars as pl

from ehrdrec.tasks.base import Task, TaskOutput
from ehrdrec.requirements import (
    Feature,
    InputRequirement,
    InputStructure,
    TaskRequirement,
)
from ehrdrec.utils import NDCATCMapper, Vocab


class MedicationSplitType(Enum):
    LAST_VISIT = auto()
    TIME = auto()


@dataclass
class MedicationSetRecommendationTaskOutput(TaskOutput):
    vocab: dict[str, Vocab]


class MedicationSetRecommendationTask(Task):
    """
    Predict the complete medication set for a visit.

    The task owns:
        - medication target construction
        - ATC conversion
        - vocab construction
        - cohort construction
        - temporal ordering
        - historical information availability
        - train / validation / test splitting

    Model-specific representation is handled by the adapter.
    """

    _requirements: set[TaskRequirement] = {
        TaskRequirement.MEDICATIONS,
        TaskRequirement.VISIT_TIMES,
    }

    def preprocess(
        self,
        raw_frames: dict[str, pl.LazyFrame],
        input_requirements: set[InputRequirement],
    ) -> MedicationSetRecommendationTaskOutput:

        atc_level = self.config.get("atc_level", 5)

        split_type = self.config.get(
            "split_type",
            MedicationSplitType.LAST_VISIT,
        )

        requested_features = {
            requirement.feature
            for requirement in input_requirements
        }

        # ============================================================
        # Medication targets
        #
        # Always required because medications define this task.
        # ============================================================

        medications_frame = self._preprocess_medications(
            raw_frames["PRESCRIPTIONS"],
            atc_level=atc_level,
        )

        # ============================================================
        # Optional model inputs
        # ============================================================

        diagnoses_frame: pl.LazyFrame | None = None

        if Feature.DIAGNOSES in requested_features:
            diagnoses_frame = self._preprocess_diagnoses(
                raw_frames["DIAGNOSES_ICD"]
            )

        procedures_frame: pl.LazyFrame | None = None

        if Feature.PROCEDURES in requested_features:
            procedures_frame = self._preprocess_procedures(
                raw_frames["PROCEDURES_ICD"]
            )

        # ============================================================
        # Global vocabularies
        #
        # Deliberately constructed BEFORE splitting.
        # ============================================================

        vocab = self._build_vocabs(
            diagnoses_frame=diagnoses_frame,
            procedures_frame=procedures_frame,
            medications_frame=medications_frame,
        )

        # ============================================================
        # Construct leakage-safe examples and split
        # ============================================================

        train, validation, test = self._split_data(
            diagnoses_frame=diagnoses_frame,
            procedures_frame=procedures_frame,
            medications_frame=medications_frame,
            admissions_frame=raw_frames["ADMISSIONS"],
            input_requirements=input_requirements,
            split_type=split_type,
        )

        return MedicationSetRecommendationTaskOutput(
            train=train,
            validation=validation,
            test=test,
            vocab=vocab,
        )
    
    def loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
        )

    # ================================================================
    # Diagnoses
    # ================================================================

    def _preprocess_diagnoses(
        self,
        frame: pl.LazyFrame,
    ) -> pl.LazyFrame:

        return (
            frame
            .group_by([
                "SUBJECT_ID",
                "HADM_ID",
            ])
            .agg(
                pl.col("ICD9_CODE")
                .sort_by([
                    "SEQ_NUM",
                    "ROW_ID",
                ])
                .drop_nulls()
                .alias("DIAGNOSES")
            )
        )

    # ================================================================
    # Procedures
    # ================================================================

    def _preprocess_procedures(
        self,
        frame: pl.LazyFrame,
    ) -> pl.LazyFrame:

        return (
            frame
            .group_by([
                "SUBJECT_ID",
                "HADM_ID",
            ])
            .agg(
                pl.col("ICD9_CODE")
                .sort_by([
                    "SEQ_NUM",
                    "ROW_ID",
                ])
                .drop_nulls()
                .alias("PROCEDURES")
            )
        )

    # ================================================================
    # Medications
    # ================================================================

    def _preprocess_medications(
        self,
        frame: pl.LazyFrame,
        atc_level: int,
    ) -> pl.LazyFrame:

        medications_frame = (
            frame
            .group_by([
                "SUBJECT_ID",
                "HADM_ID",
            ])
            .agg(
                pl.struct([
                    pl.col("STARTDATE"),
                    pl.col("ENDDATE"),
                    pl.col("DRUG_TYPE"),
                    pl.col("DRUG"),
                    pl.col("DRUG_NAME_POE"),
                    pl.col("DRUG_NAME_GENERIC"),
                    pl.col("FORMULARY_DRUG_CD"),
                    pl.col("GSN"),
                    pl.col("NDC"),
                    pl.col("PROD_STRENGTH"),
                    pl.col("DOSE_VAL_RX"),
                    pl.col("DOSE_UNIT_RX"),
                    pl.col("FORM_VAL_DISP"),
                    pl.col("FORM_UNIT_DISP"),
                    pl.col("ROUTE"),
                ])
                .sort_by([
                    "STARTDATE",
                    "ENDDATE",
                    "DRUG_NAME_GENERIC",
                    "NDC",
                    "ROW_ID",
                ])
                .alias("MEDICATIONS")
            )
        )

        return self._convert_ndc_to_atc(
            medications_frame,
            atc_level=atc_level,
        )

    # ================================================================
    # NDC -> ATC
    # ================================================================

    def _convert_ndc_to_atc(
        self,
        medications_frame: pl.LazyFrame,
        atc_level: int,
    ) -> pl.LazyFrame:

        mapper = NDCATCMapper.from_file(
            path=self.config[
                "ndc_atc_mapping_file"
            ]
        )

        ndc_cache: dict[str, list[str]] = {}

        def cached_ndc_to_atc(
            ndc: str,
        ) -> list[str]:

            if ndc in ndc_cache:
                return ndc_cache[ndc]

            mapped = mapper.ndc_to_atc(
                ndc,
                atc_level=atc_level,
            )

            if not mapped or not mapped.atc_codes:
                result = ["UNK"]
            else:
                result = list(mapped.atc_codes)

            ndc_cache[ndc] = result

            return result

        def add_atc_to_medications(meds):
            if meds is None:
                return []

            meds = (
                meds.to_list()
                if hasattr(meds, "to_list")
                else list(meds)
            )

            result = []

            for med in meds:
                if med is None:
                    continue

                med = dict(med)

                ndc = med.get("NDC")

                if (
                    ndc is None
                    or str(ndc).strip() in {"", "0"}
                ):
                    med["ATC_CODES"] = ["UNK"]

                else:
                    med["ATC_CODES"] = (
                        cached_ndc_to_atc(
                            str(ndc).strip()
                        )
                    )

                result.append(med)

            return result

        schema = medications_frame.collect_schema()

        medications_dtype = schema["MEDICATIONS"]
        medication_struct_dtype = (
            medications_dtype.inner
        )

        extended_struct_dtype = pl.Struct([
            *medication_struct_dtype.fields,

            pl.Field(
                "ATC_CODES",
                pl.List(pl.String),
            ),
        ])

        return medications_frame.with_columns(
            pl.col("MEDICATIONS")
            .map_elements(
                add_atc_to_medications,
                return_dtype=pl.List(
                    extended_struct_dtype
                ),
            )
        )

    # ================================================================
    # Vocab
    # ================================================================

    def _build_vocabs(
        self,
        diagnoses_frame: pl.LazyFrame | None,
        procedures_frame: pl.LazyFrame | None,
        medications_frame: pl.LazyFrame,
    ) -> dict[str, Vocab]:

        vocab: dict[str, Vocab] = {}

        # Diagnosis vocab only exists if some model input requires it.
        if diagnoses_frame is not None:

            diagnosis_tokens = (
                diagnoses_frame
                .select(
                    pl.col("DIAGNOSES")
                    .explode()
                    .alias("token")
                )
            )

            vocab["diagnoses"] = Vocab.from_tokens(
                diagnosis_tokens
            )

        # Procedure vocab only exists if requested.
        if procedures_frame is not None:

            procedure_tokens = (
                procedures_frame
                .select(
                    pl.col("PROCEDURES")
                    .explode()
                    .alias("token")
                )
            )

            vocab["procedures"] = Vocab.from_tokens(
                procedure_tokens
            )

        # Medication vocab always exists because medications
        # constitute the target of this task.
        medication_tokens = (
            medications_frame
            .select(
                pl.col("MEDICATIONS")
                .explode()
                .alias("medication")
            )
            .select(
                pl.col("medication")
                .struct.field("ATC_CODES")
                .explode()
                .alias("token")
            )
        )

        vocab["medications"] = Vocab.from_tokens(
            medication_tokens
        )

        return vocab

    # ================================================================
    # Prediction example construction + splitting
    # ================================================================

    def _split_data(
        self,
        diagnoses_frame: pl.LazyFrame | None,
        procedures_frame: pl.LazyFrame | None,
        medications_frame: pl.LazyFrame,
        admissions_frame: pl.LazyFrame,
        input_requirements: set[InputRequirement],
        split_type: MedicationSplitType,
    ) -> tuple[
        pl.LazyFrame,
        pl.LazyFrame,
        pl.LazyFrame,
    ]:

        keys = [
            "SUBJECT_ID",
            "HADM_ID",
        ]

        requested_features = {
            requirement.feature
            for requirement in input_requirements
        }

        sequence_features = {
            requirement.feature
            for requirement in input_requirements
            if (
                requirement.structure
                == InputStructure.VISIT_SEQUENCE
            )
        }

        # Medication history is historical by definition.
        #
        # Any VISIT_SEQUENCE also requires previous visits.
        needs_history = (
            Feature.MEDICATION_HISTORY
            in requested_features
            or bool(sequence_features)
        )

        # ============================================================
        # Define task cohort
        #
        # This is based ONLY on information intrinsically required
        # by the task:
        #
        #   admission + medication target
        #
        # Optional model features must NOT change the cohort.
        # ============================================================

        visits = (
            admissions_frame
            .select(
                "SUBJECT_ID",
                "HADM_ID",
                "ADMITTIME",
            )
            .join(
                medications_frame,
                on=keys,
                how="inner",
            )
        )

        # ============================================================
        # Add optional model features
        #
        # LEFT joins are deliberate.
        #
        # A model requesting diagnoses should not produce a different
        # patient cohort from a model that doesn't request diagnoses.
        # ============================================================

        if diagnoses_frame is not None:
            visits = visits.join(
                diagnoses_frame,
                on=keys,
                how="left",
            )

        if procedures_frame is not None:
            visits = visits.join(
                procedures_frame,
                on=keys,
                how="left",
            )

        # ============================================================
        # Current visit representation
        # ============================================================

        current_fields = [
            pl.col("HADM_ID"),
            pl.col("ADMITTIME"),
            pl.col("MEDICATIONS"),
        ]

        if diagnoses_frame is not None:
            current_fields.append(
                pl.col("DIAGNOSES")
            )

        if procedures_frame is not None:
            current_fields.append(
                pl.col("PROCEDURES")
            )

        aggregation_expressions = [
            pl.struct(current_fields)
            .sort_by([
                "ADMITTIME",
                "HADM_ID",
            ])
            .alias("_CURRENT_VISITS")
        ]

        # ============================================================
        # Historical representation
        #
        # Only expose historical features actually requested.
        # ============================================================

        if needs_history:

            history_fields = [
                pl.col("HADM_ID"),
                pl.col("ADMITTIME"),
            ]

            if (
                Feature.DIAGNOSES
                in sequence_features
            ):
                history_fields.append(
                    pl.col("DIAGNOSES")
                )

            if (
                Feature.PROCEDURES
                in sequence_features
            ):
                history_fields.append(
                    pl.col("PROCEDURES")
                )

            if (
                Feature.MEDICATION_HISTORY
                in requested_features
            ):
                history_fields.append(
                    pl.col("MEDICATIONS")
                )

            aggregation_expressions.append(
                pl.struct(history_fields)
                .sort_by([
                    "ADMITTIME",
                    "HADM_ID",
                ])
                .alias("_HISTORY_VISITS")
            )

        # ============================================================
        # Build chronologically ordered patient sequences
        # ============================================================

        patient_sequences = (
            visits
            .group_by("SUBJECT_ID")
            .agg(
                aggregation_expressions
            )
        )

        # LAST_VISIT requires at least:
        #
        #   one train
        #   one validation
        #   one test
        #
        # Importantly, this counts eligible task visits, not raw
        # admissions.
        if split_type == MedicationSplitType.LAST_VISIT:

            patient_sequences = (
                patient_sequences
                .filter(
                    pl.col("_CURRENT_VISITS")
                    .list.len()
                    >= 3
                )
            )

        # ============================================================
        # One row per prediction visit
        # ============================================================

        examples = (
            patient_sequences
            .with_columns(
                pl.col("_CURRENT_VISITS")
                .list.len()
                .alias("_N_VISITS"),

                pl.int_ranges(
                    0,
                    pl.col("_CURRENT_VISITS")
                    .list.len(),
                )
                .alias("_VISIT_INDEX"),
            )
            .explode("_VISIT_INDEX")
            .with_columns(
                pl.col("_CURRENT_VISITS")
                .list.get(
                    pl.col("_VISIT_INDEX")
                )
                .alias("_CURRENT_VISIT")
            )
        )

        # Strictly preceding visits only.
        if needs_history:

            examples = examples.with_columns(
                pl.col("_HISTORY_VISITS")
                .list.slice(
                    0,
                    pl.col("_VISIT_INDEX"),
                )
                .alias("PREVIOUS_VISITS")
            )

        examples = (
            examples
            .unnest("_CURRENT_VISIT")
            .rename({
                "MEDICATIONS":
                    "TARGET_MEDICATIONS",
            })
        )

        if diagnoses_frame is not None:
            examples = examples.rename({
                "DIAGNOSES":
                    "CURRENT_DIAGNOSES",
            })

        if procedures_frame is not None:
            examples = examples.rename({
                "PROCEDURES":
                    "CURRENT_PROCEDURES",
            })

        # ============================================================
        # Split assignment
        # ============================================================

        if split_type == MedicationSplitType.LAST_VISIT:

            examples = examples.with_columns(
                pl.when(
                    pl.col("_VISIT_INDEX")
                    == pl.col("_N_VISITS") - 1
                )
                .then(pl.lit("test"))

                .when(
                    pl.col("_VISIT_INDEX")
                    == pl.col("_N_VISITS") - 2
                )
                .then(pl.lit("validation"))

                .otherwise(
                    pl.lit("train")
                )
                .alias("_SPLIT")
            )

        elif split_type == MedicationSplitType.TIME:

            examples = (
                examples
                .sort([
                    "ADMITTIME",
                    "SUBJECT_ID",
                    "HADM_ID",
                ])
                .with_row_index(
                    "_GLOBAL_INDEX"
                )
                .with_columns(
                    pl.len()
                    .alias("_N_EXAMPLES")
                )
                .with_columns(
                    pl.when(
                        pl.col("_GLOBAL_INDEX")
                        < (
                            pl.col("_N_EXAMPLES")
                            * 0.70
                        ).floor()
                    )
                    .then(pl.lit("train"))

                    .when(
                        pl.col("_GLOBAL_INDEX")
                        < (
                            pl.col("_N_EXAMPLES")
                            * 0.85
                        ).floor()
                    )
                    .then(
                        pl.lit("validation")
                    )

                    .otherwise(
                        pl.lit("test")
                    )
                    .alias("_SPLIT")
                )
            )

        else:
            raise ValueError(
                f"Unsupported split type: "
                f"{split_type}"
            )

        # ============================================================
        # Public task output
        # ============================================================

        output_columns = [
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
        ]

        if diagnoses_frame is not None:
            output_columns.append(
                "CURRENT_DIAGNOSES"
            )

        if procedures_frame is not None:
            output_columns.append(
                "CURRENT_PROCEDURES"
            )

        if needs_history:
            output_columns.append(
                "PREVIOUS_VISITS"
            )

        output_columns.append(
            "TARGET_MEDICATIONS"
        )

        def select_split(
            split: str,
        ) -> pl.LazyFrame:

            return (
                examples
                .filter(
                    pl.col("_SPLIT")
                    == split
                )
                .select(
                    output_columns
                )
                .sort([
                    "SUBJECT_ID",
                    "ADMITTIME",
                    "HADM_ID",
                ])
            )

        return (
            select_split("train"),
            select_split("validation"),
            select_split("test"),
        )