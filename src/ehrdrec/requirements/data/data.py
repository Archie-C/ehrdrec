from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass

class DataRequirement(Enum):
    """
    Semantic raw data requirements.

    Dataset loaders translate these into physical tables/files.
    """

    DIAGNOSES = auto()
    PROCEDURES = auto()
    MEDICATIONS = auto()
    VISIT_TIMES = auto()


@dataclass(frozen=True)
class DataRequest:
    requirements: frozenset[DataRequirement]