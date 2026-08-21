from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset

from ehrdrec.contracts.adapters import (
    BatchFieldSpec,
    BatchTargetSpec,
)
from ehrdrec.requirements.model import (
    InputStructure,
    Representation,
)


class EHRDataset(Dataset):
    """
    Generic dataset for adapted EHRDRec data.

    This class has no knowledge of:
        - tasks
        - models
        - features
        - representations

    It simply exposes one adapted row at a time.
    """

    def __init__(
        self,
        frame: pl.LazyFrame | pl.DataFrame,
    ) -> None:

        if isinstance(
            frame,
            pl.LazyFrame,
        ):
            frame = frame.collect()

        self.frame = frame

    def __len__(
        self,
    ) -> int:

        return self.frame.height

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.frame.row(
            index,
            named=True,
        )

        if "EXAMPLE_ID" not in row:
            if "SUBJECT_ID" in row and "HADM_ID" in row:
                row["EXAMPLE_ID"] = (
                    f"{row['SUBJECT_ID']}:{row['HADM_ID']}"
                )
            else:
                # Stable for a fixed, deterministically ordered processed split.
                row["EXAMPLE_ID"] = f"example_{index:08d}"

        return row


@dataclass
class EHRBatch:
    inputs: dict[str, Any]
    targets: torch.Tensor
    metadata: dict[str, Any]

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        inputs = self.__dict__.get(
            "inputs",
            {},
        )

        if name in inputs:
            return inputs[name]

        raise AttributeError(
            f"{type(self).__name__!s} "
            f"has no attribute {name!r}"
        )

    def to(
        self,
        device: torch.device | str,
    ) -> "EHRBatch":
        return EHRBatch(
            inputs={
                name: self._move_to_device(value, device)
                for name, value in self.inputs.items()
            },
            targets=self.targets.to(device),
            metadata=self.metadata,
        )

    @classmethod
    def _move_to_device(
        cls,
        value: Any,
        device: torch.device | str,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)

        if isinstance(value, list):
            return [
                cls._move_to_device(item, device)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls._move_to_device(item, device)
                for item in value
            )

        if isinstance(value, dict):
            return {
                key: cls._move_to_device(item, device)
                for key, item in value.items()
            }

        return value


class EHRBatchCollator:
    """
    Generic collator for all EHRDRec models.

    The adapter provides BatchFieldSpecs describing how each compact
    field should be represented.

    This class turns the compact vocabulary-ID representation into
    the actual tensors required by the model.
    """

    def __init__(
        self,
        fields: dict[
            str,
            BatchFieldSpec,
        ],
        target: BatchTargetSpec,
    ) -> None:

        self.fields = fields

        self.target = target

    # ================================================================
    # Public API
    # ================================================================

    def __call__(
        self,
        examples: list[
            dict[str, Any]
        ],
    ) -> EHRBatch:

        inputs: dict[
            str,
            Any,
        ] = {}

        # ------------------------------------------------------------
        # Model inputs
        # ------------------------------------------------------------

        for name, specification in self.fields.items():

            values = [
                example[name]
                for example in examples
            ]

            inputs[name] = (
                self._collate_feature(
                    values=values,
                    specification=specification,
                )
            )

        # ------------------------------------------------------------
        # Targets
        # ------------------------------------------------------------

        target_values = [
            example[self.target.name]
            for example in examples
        ]

        targets = self._collate_target(
            values=target_values,
            specification=self.target,
        )

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata = {}

        for name in (
            "EXAMPLE_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
        ):

            if name in examples[0]:

                metadata[name] = [
                    example[name]
                    for example in examples
                ]

        return EHRBatch(
            inputs=inputs,
            targets=targets,
            metadata=metadata,
        )

    # ================================================================
    # Feature collation
    # ================================================================

    def _collate_feature(
        self,
        values: list[Any],
        specification: BatchFieldSpec,
    ) -> Any:

        requirement = (
            specification.requirement
        )

        representation = (
            requirement.representation
        )

        structure = (
            requirement.structure
        )

        # ------------------------------------------------------------
        # MULTI_HOT + VISIT_SEQUENCE
        #
        # RETAIN currently uses this.
        #
        # Result:
        #
        # [
        #     Tensor[T1, vocab_size],
        #     Tensor[T2, vocab_size],
        #     ...
        # ]
        #
        # We deliberately don't pad yet because RETAIN accepts
        # variable-length patient tensors.
        # ------------------------------------------------------------

        if (
            representation
            == Representation.MULTI_HOT
            and structure
            == InputStructure.VISIT_SEQUENCE
        ):

            vocab_size = (
                self._require_vocab_size(
                    specification
                )
            )

            return [
                self._multihot_sequence(
                    sequence=value,
                    vocab_size=vocab_size,
                )
                for value in values
            ]

        # ------------------------------------------------------------
        # MULTI_HOT + FLAT
        #
        # Result:
        #
        # Tensor[B, vocab_size]
        # ------------------------------------------------------------

        if (
            representation
            == Representation.MULTI_HOT
            and structure
            == InputStructure.FLAT
        ):

            vocab_size = (
                self._require_vocab_size(
                    specification
                )
            )

            return torch.stack([
                self._multihot(
                    ids=value,
                    vocab_size=vocab_size,
                )
                for value in values
            ])

        # ------------------------------------------------------------
        # CODE_LIST + FLAT
        #
        # Variable-length list of code tensors.
        # ------------------------------------------------------------

        if (
            representation
            == Representation.CODE_LIST
            and structure
            == InputStructure.FLAT
        ):

            return [
                self._code_tensor(
                    value
                )
                for value in values
            ]

        # ------------------------------------------------------------
        # CODE_LIST + VISIT_SEQUENCE
        #
        # Result:
        #
        # [
        #     [
        #         Tensor[num_codes_visit_1],
        #         Tensor[num_codes_visit_2],
        #         ...
        #     ],
        #     ...
        # ]
        # ------------------------------------------------------------

        if (
            representation
            == Representation.CODE_LIST
            and structure
            == InputStructure.VISIT_SEQUENCE
        ):

            return [
                [
                    self._code_tensor(
                        visit
                    )
                    for visit in self._as_list(
                        sequence
                    )
                ]
                for sequence in values
            ]

        raise NotImplementedError(
            "EHRBatchCollator does not support "
            f"{representation.name} + "
            f"{structure.name}."
        )

    # ================================================================
    # Target collation
    # ================================================================

    def _collate_target(
        self,
        values: list[Any],
        specification: BatchTargetSpec,
    ) -> torch.Tensor:

        if (
            specification.representation
            == Representation.MULTI_HOT
        ):

            if specification.vocab_size is None:
                raise ValueError(
                    "MULTI_HOT target representation "
                    "requires vocab_size."
                )

            return torch.stack([
                self._multihot(
                    ids=value,
                    vocab_size=(
                        specification.vocab_size
                    ),
                )
                for value in values
            ])

        raise NotImplementedError(
            "EHRBatchCollator does not yet support "
            "target representation "
            f"{specification.representation.name}."
        )

    # ================================================================
    # Multi-hot
    # ================================================================

    @staticmethod
    def _multihot(
        ids,
        vocab_size: int,
    ) -> torch.Tensor:

        vector = torch.zeros(
            vocab_size,
            dtype=torch.float32,
        )

        ids = EHRBatchCollator._as_list(
            ids
        )

        valid_ids = [
            int(i)
            for i in ids
            if (
                i is not None
                and 0 <= int(i) < vocab_size
            )
        ]

        if valid_ids:

            index = torch.tensor(
                valid_ids,
                dtype=torch.long,
            )

            vector[index] = 1.0

        return vector

    @classmethod
    def _multihot_sequence(
        cls,
        sequence,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Convert:

            [
                [1, 4, 7],
                [2, 8],
                [3, 9, 10],
            ]

        into:

            Tensor[
                num_visits,
                vocab_size
            ]
        """

        visits = cls._as_list(
            sequence
        )

        if not visits:

            return torch.zeros(
                (
                    0,
                    vocab_size,
                ),
                dtype=torch.float32,
            )

        return torch.stack([
            cls._multihot(
                ids=visit,
                vocab_size=vocab_size,
            )
            for visit in visits
        ])

    # ================================================================
    # Code list
    # ================================================================

    @classmethod
    def _code_tensor(
        cls,
        ids,
    ) -> torch.Tensor:

        ids = cls._as_list(
            ids
        )

        return torch.tensor(
            [
                int(i)
                for i in ids
                if i is not None
            ],
            dtype=torch.long,
        )

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _as_list(
        value,
    ) -> list:

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
    def _require_vocab_size(
        specification: BatchFieldSpec,
    ) -> int:

        if specification.vocab_size is None:

            raise ValueError(
                f"Field '{specification.name}' "
                "requires a vocabulary size for "
                f"{specification.requirement.representation.name}."
            )

        return specification.vocab_size