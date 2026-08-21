from __future__ import annotations

from typing import Any

import polars as pl

from ehrdrec.contracts.adapters import (
    AdapterOutput,
    BatchFieldSpec,
    BatchTargetSpec,
)
from ehrdrec.requirements.model import (
    Feature,
    InputRequirement,
    InputStructure,
    Representation,
)
from ehrdrec.tasks.medication_set_recommendation.task import (
    MedicationSetRecommendationTaskOutput,
)
from ehrdrec.utils import (
    ReservedId,
    Vocab,
)


class MedicationSetRecommendationAdapter:
    """
    Adapter for MedicationSetRecommendationTask.

    The Task owns:
        - cohort construction
        - splitting
        - temporal validity
        - leakage prevention
        - current vs previous visit semantics
        - target construction

    The Adapter owns:
        - vocabulary encoding
        - selecting the information requested by the model
        - producing a compact model-ready representation

    Dense tensor representations such as MULTI_HOT are NOT produced
    here. They are produced by the generic EHRBatchCollator when a
    batch is created.

    Currently supported input requirements:
        DIAGNOSES
            MULTI_HOT + VISIT_SEQUENCE

        PROCEDURES
            MULTI_HOT + VISIT_SEQUENCE

    This is sufficient for RETAIN.
    """

    _supported_requirements: set[InputRequirement] = {
        InputRequirement(
            feature=Feature.DIAGNOSES,
            representation=Representation.MULTI_HOT,
            structure=InputStructure.VISIT_SEQUENCE,
        ),
        InputRequirement(
            feature=Feature.PROCEDURES,
            representation=Representation.MULTI_HOT,
            structure=InputStructure.VISIT_SEQUENCE,
        ),
    }

    def __init__(
        self,
        task_output: MedicationSetRecommendationTaskOutput,
        input_requirements: set[InputRequirement],
    ) -> None:
        self.task_output = task_output

        self.input_requirements = set(
            input_requirements
        )

        self._validate_requirements()

    # ================================================================
    # Public API
    # ================================================================

    def adapt(self) -> AdapterOutput:
        """
        Adapt all task splits.

        The returned frames contain compact integer vocabulary IDs.

        For RETAIN:

            diagnoses:
                List[List[int]]

            procedures:
                List[List[int]]

            targets:
                List[int]

        Dense multi-hot vectors are generated later by the collator.
        """

        fields = self._build_field_specs()

        target = BatchTargetSpec(
            name="targets",
            representation=Representation.MULTI_HOT,
            vocab_size=self._get_vocab(
                "medications"
            ).vocab_size,
        )

        return AdapterOutput(
            train=self.transform(
                self.task_output.train
            ),
            validation=self.transform(
                self.task_output.validation
            ),
            test=self.transform(
                self.task_output.test
            ),
            fields=fields,
            target=target,
        )

    def transform(
        self,
        frame: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """
        Adapt one task split into compact model-ready values.
        """

        expressions: list[pl.Expr] = [
            pl.col("SUBJECT_ID"),
            pl.col("HADM_ID"),
            pl.col("ADMITTIME"),
        ]

        # ------------------------------------------------------------
        # Diagnoses
        # ------------------------------------------------------------

        if self._requires(
            feature=Feature.DIAGNOSES,
            representation=Representation.MULTI_HOT,
            structure=InputStructure.VISIT_SEQUENCE,
        ):
            expressions.append(
                self._encode_visit_sequence(
                    current_col="CURRENT_DIAGNOSES",
                    history_field="DIAGNOSES",
                    vocab=self._get_vocab(
                        "diagnoses"
                    ),
                )
                .alias("diagnoses")
            )

        # ------------------------------------------------------------
        # Procedures
        # ------------------------------------------------------------

        if self._requires(
            feature=Feature.PROCEDURES,
            representation=Representation.MULTI_HOT,
            structure=InputStructure.VISIT_SEQUENCE,
        ):
            expressions.append(
                self._encode_visit_sequence(
                    current_col="CURRENT_PROCEDURES",
                    history_field="PROCEDURES",
                    vocab=self._get_vocab(
                        "procedures"
                    ),
                )
                .alias("procedures")
            )

        # ------------------------------------------------------------
        # Medication target
        #
        # Current medications NEVER become model inputs here.
        # ------------------------------------------------------------

        expressions.append(
            self._encode_medication_target()
            .alias("targets")
        )

        return frame.select(expressions)

    # ================================================================
    # Batch specifications
    # ================================================================

    def _build_field_specs(
        self,
    ) -> dict[str, BatchFieldSpec]:

        fields: dict[str, BatchFieldSpec] = {}

        for requirement in self.input_requirements:

            if requirement.feature == Feature.DIAGNOSES:

                fields["diagnoses"] = BatchFieldSpec(
                    name="diagnoses",
                    requirement=requirement,
                    vocab_size=self._get_vocab(
                        "diagnoses"
                    ).vocab_size,
                )

            elif requirement.feature == Feature.PROCEDURES:

                fields["procedures"] = BatchFieldSpec(
                    name="procedures",
                    requirement=requirement,
                    vocab_size=self._get_vocab(
                        "procedures"
                    ).vocab_size,
                )

            else:
                raise NotImplementedError(
                    "MedicationSetRecommendationAdapter does "
                    f"not yet support feature "
                    f"{requirement.feature.name}."
                )

        return fields

    # ================================================================
    # Requirement handling
    # ================================================================

    def _validate_requirements(
        self,
    ) -> None:

        unsupported = (
            self.input_requirements
            - self._supported_requirements
        )

        if not unsupported:
            return

        formatted = "\n".join(
            f"    {requirement}"
            for requirement in sorted(
                unsupported,
                key=str,
            )
        )

        raise NotImplementedError(
            "MedicationSetRecommendationAdapter does not "
            "yet support the following input requirements:\n"
            f"{formatted}"
        )

    def _requires(
        self,
        feature: Feature,
        representation: Representation,
        structure: InputStructure,
    ) -> bool:

        requirement = InputRequirement(
            feature=feature,
            representation=representation,
            structure=structure,
        )

        return requirement in self.input_requirements

    # ================================================================
    # Visit sequence encoding
    # ================================================================

    def _encode_visit_sequence(
        self,
        current_col: str,
        history_field: str,
        vocab: Vocab,
    ) -> pl.Expr:
        """
        Encode a chronological visit sequence using vocabulary IDs.

        Example:

            PREVIOUS_VISITS:
                visit 1 diagnoses = ["4019", "25000"]
                visit 2 diagnoses = ["41401"]

            CURRENT_DIAGNOSES:
                ["42731"]

        becomes:

            [
                [12, 45],
                [83],
                [102],
            ]

        The current visit is appended after all historical visits.

        The Task has already guaranteed that PREVIOUS_VISITS contains
        only chronologically valid historical information.
        """

        def encode_sequence(
            row: dict[str, Any],
        ) -> list[list[int]]:

            sequence: list[list[int]] = []

            # --------------------------------------------------------
            # Previous visits
            # --------------------------------------------------------

            previous_visits = self._as_list(
                row.get("PREVIOUS_VISITS")
            )

            for visit in previous_visits:

                if visit is None:
                    continue

                tokens = self._struct_field(
                    visit,
                    history_field,
                )

                sequence.append(
                    self._encode_tokens(
                        tokens=tokens,
                        vocab=vocab,
                    )
                )

            # --------------------------------------------------------
            # Current visit
            # --------------------------------------------------------

            current_tokens = row.get(
                current_col
            )

            sequence.append(
                self._encode_tokens(
                    tokens=current_tokens,
                    vocab=vocab,
                )
            )

            return sequence

        return (
            pl.struct([
                "PREVIOUS_VISITS",
                current_col,
            ])
            .map_elements(
                encode_sequence,
                return_dtype=pl.List(
                    pl.List(
                        pl.Int64
                    )
                ),
            )
        )

    # ================================================================
    # Medication target
    # ================================================================

    def _encode_medication_target(
        self,
    ) -> pl.Expr:
        """
        Convert TARGET_MEDICATIONS into medication vocabulary IDs.

        Example:

            TARGET_MEDICATIONS:
                [
                    {ATC_CODES: ["A10BA02"]},
                    {ATC_CODES: ["C09AA05"]},
                ]

        becomes:

            [13, 82]

        Multi-hot conversion happens later in the collator.
        """

        vocab = self._get_vocab(
            "medications"
        )

        unknown_id = int(
            ReservedId.UNK
        )

        def encode_target(
            medications,
        ) -> list[int]:

            medications = self._as_list(
                medications
            )

            ids: list[int] = []

            for medication in medications:

                if medication is None:
                    continue

                atc_codes = self._struct_field(
                    medication,
                    "ATC_CODES",
                )

                for code in self._as_list(
                    atc_codes
                ):

                    if code is None:
                        continue

                    token_id = (
                        vocab.token_to_id.get(
                            str(code),
                            unknown_id,
                        )
                    )

                    ids.append(
                        token_id
                    )

            # Remove duplicates while preserving deterministic order.
            return list(
                dict.fromkeys(ids)
            )

        return (
            pl.col("TARGET_MEDICATIONS")
            .map_elements(
                encode_target,
                return_dtype=pl.List(
                    pl.Int64
                ),
            )
        )

    # ================================================================
    # Encoding helpers
    # ================================================================

    @staticmethod
    def _encode_tokens(
        tokens,
        vocab: Vocab,
    ) -> list[int]:
        """
        Convert a collection of string tokens into vocabulary IDs.
        """

        values = MedicationSetRecommendationAdapter._as_list(
            tokens
        )

        values = [
            str(value)
            for value in values
            if value is not None
        ]

        if not values:
            return []

        return vocab.encode_list(
            values
        )

    @staticmethod
    def _as_list(
        value,
    ) -> list:
        """
        Normalize Polars/list-like values to Python lists.
        """

        if value is None:
            return []

        if hasattr(
            value,
            "to_list",
        ):
            return value.to_list()

        if isinstance(
            value,
            (list, tuple),
        ):
            return list(value)

        return [value]

    @staticmethod
    def _struct_field(
        value,
        field: str,
    ):
        """
        Read a field from a Polars struct representation.
        """

        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):
            return value.get(
                field
            )

        try:
            return value[field]

        except (
            KeyError,
            TypeError,
            IndexError,
        ):
            return None

    # ================================================================
    # Resources
    # ================================================================

    def _get_vocab(
        self,
        name: str,
    ) -> Vocab:

        try:
            return self.task_output.vocab[
                name
            ]

        except KeyError as exc:
            raise ValueError(
                f"Required '{name}' vocabulary is "
                "not present in the task output."
            ) from exc