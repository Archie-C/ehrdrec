from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ehrdrec.requirements.model import (
    InputRequirement,
    Representation,
)


@dataclass(frozen=True)
class BatchFieldSpec:
    """
    Describes how one adapted input field should be collated.

    The adapter stores values compactly as vocabulary IDs.
    The collator uses this specification to construct the actual
    representation requested by the model.
    """

    name: str

    requirement: InputRequirement

    # Required for representations such as MULTI_HOT.
    # Can be None for features without a vocabulary.
    vocab_size: int | None = None


@dataclass(frozen=True)
class BatchTargetSpec:
    """
    Describes how the prediction target should be collated.
    """

    name: str

    representation: Representation

    vocab_size: int | None = None


@dataclass(frozen=True)
class AdapterOutput:
    """
    Standard output produced by an EHRDRec task adapter.

    Every model can therefore use the same Dataset and Collator.
    """

    train: pl.LazyFrame
    validation: pl.LazyFrame
    test: pl.LazyFrame

    fields: dict[str, BatchFieldSpec]

    target: BatchTargetSpec