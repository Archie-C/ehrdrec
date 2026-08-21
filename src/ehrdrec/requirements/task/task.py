from __future__ import annotations

from enum import Enum, auto

class TaskRequirement(Enum):
    """
    Semantic information inherently required to define the task.

    These requirements exist regardless of which model is used.
    """

    DIAGNOSES = auto()
    PROCEDURES = auto()
    MEDICATIONS = auto()
    VISIT_TIMES = auto()