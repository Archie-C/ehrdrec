from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import polars as pl

from ehrdrec.requirements import (
    DataRequest,
    DataRequirement,
    Feature,
    InputRequirement,
    InputStructure,
    ModelRequirement,
    TaskRequirement,
)


@dataclass
class TaskOutput:
    """
    Base output produced by a Task.

    Concrete tasks should subclass this with their own split
    structure and resources.
    """
    train: pl.LazyFrame
    validation: pl.LazyFrame
    test: pl.LazyFrame


TASK_DATA_REQUIREMENTS: dict[
    TaskRequirement,
    set[DataRequirement],
] = {
    TaskRequirement.DIAGNOSES: {
        DataRequirement.DIAGNOSES,
    },
    TaskRequirement.PROCEDURES: {
        DataRequirement.PROCEDURES,
    },
    TaskRequirement.MEDICATIONS: {
        DataRequirement.MEDICATIONS,
    },
    TaskRequirement.VISIT_TIMES: {
        DataRequirement.VISIT_TIMES,
    },
}


FEATURE_DATA_REQUIREMENTS: dict[
    Feature,
    set[DataRequirement],
] = {
    Feature.DIAGNOSES: {
        DataRequirement.DIAGNOSES,
    },
    Feature.PROCEDURES: {
        DataRequirement.PROCEDURES,
    },
    Feature.MEDICATION_HISTORY: {
        DataRequirement.MEDICATIONS,
        DataRequirement.VISIT_TIMES,
    },
    Feature.VISIT_TIMES: {
        DataRequirement.VISIT_TIMES,
    },
}


STRUCTURE_DATA_REQUIREMENTS: dict[
    InputStructure,
    set[DataRequirement],
] = {
    InputStructure.FLAT: set(),

    InputStructure.VISIT_SEQUENCE: {
        DataRequirement.VISIT_TIMES,
    },
}


MODEL_DATA_REQUIREMENTS: dict[
    ModelRequirement,
    set[DataRequirement],
] = {
    ModelRequirement.EHR_MEDICATION_GRAPH: {
        DataRequirement.MEDICATIONS,
    },

    ModelRequirement.DDI_GRAPH: set(),
    ModelRequirement.MOLECULAR_GRAPHS: set(),
    ModelRequirement.MEDICATION_MOLECULE_PROJECTION: set(),
    ModelRequirement.MEDICATION_SUBSTRUCTURE_MATRIX: set(),
    ModelRequirement.SUBSTRUCTURE_GRAPHS: set(),

    ModelRequirement.DETERMINISTIC_ORDERING: set(),
}


class Task(ABC):
    """
    Base class for prediction tasks.

    A Task owns:
        - task semantics;
        - cohort construction;
        - preprocessing;
        - splitting;
        - leakage prevention;
        - construction of model-agnostic prediction examples.

    It does not own:
        - dataset-specific file loading;
        - model-specific tensor representation;
        - model architecture.
    """

    _requirements: set[TaskRequirement] = set()
    version: str = "1.0"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}

    @property
    def requirements(self) -> set[TaskRequirement]:
        return set(self._requirements)

    def get_resolved_config(self) -> dict[str, Any]:
        return dict(self.config)

    def get_data_request(
        self,
        input_requirements: set[InputRequirement],
        model_requirements: set[ModelRequirement],
    ) -> DataRequest:
        """
        Determine which semantic raw data must be loaded.

        This combines:
            1. data fundamentally required by the Task;
            2. features required by the Model;
            3. structural requirements that imply additional data;
            4. model resources that depend on raw EHR data.

        Representation does not affect loading.
        """

        required_data: set[DataRequirement] = set()

        # Task requirements
        for requirement in self._requirements:
            required_data.update(
                TASK_DATA_REQUIREMENTS[requirement]
            )

        # Model input requirements
        for requirement in input_requirements:
            required_data.update(
                FEATURE_DATA_REQUIREMENTS[
                    requirement.feature
                ]
            )

            required_data.update(
                STRUCTURE_DATA_REQUIREMENTS[
                    requirement.structure
                ]
            )

        # Additional model requirements which require EHR data
        for requirement in model_requirements:
            required_data.update(
                MODEL_DATA_REQUIREMENTS[
                    requirement
                ]
            )

        return DataRequest(
            requirements=frozenset(required_data)
        )

    @abstractmethod
    def preprocess(
        self,
        raw_frames: dict[str, pl.LazyFrame],
        input_requirements: set[InputRequirement],
        model_requirements: set[ModelRequirement],
    ) -> TaskOutput:
        """
        Convert raw dataset frames into leakage-safe,
        model-independent prediction examples.
        """
        raise NotImplementedError