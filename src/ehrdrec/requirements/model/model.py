from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

# ============================================================
# Model input requirements
# ============================================================

class Feature(Enum):
    DIAGNOSES = auto()
    PROCEDURES = auto()
    MEDICATION_HISTORY = auto()
    VISIT_TIMES = auto()


class Representation(Enum):
    CODE_LIST = auto()
    MULTI_HOT = auto()


class InputStructure(Enum):
    FLAT = auto()
    VISIT_SEQUENCE = auto()


@dataclass(frozen=True)
class InputRequirement:
    feature: Feature
    representation: Representation
    structure: InputStructure


# ============================================================
# Other model requirements
# ============================================================

class ModelRequirement(Enum):
    EHR_MEDICATION_GRAPH = auto()
    DDI_GRAPH = auto()
    MOLECULAR_GRAPHS = auto()
    MEDICATION_MOLECULE_PROJECTION = auto()
    MEDICATION_SUBSTRUCTURE_MATRIX = auto()
    SUBSTRUCTURE_GRAPHS = auto()

    DETERMINISTIC_ORDERING = auto()