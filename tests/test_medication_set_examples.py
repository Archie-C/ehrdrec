from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from ehrdrec.tasks.medication_set import (
    CurrentVisitContext,
    MedicationSetExample,
    ObservedVisit,
)


def _current_context() -> CurrentVisitContext:
    return CurrentVisitContext(
        visit_id="visit-3",
        diagnoses=(9, 2),
        procedures=(7,),
        occurred_at=datetime(2020, 1, 3),
    )


def test_example_separates_history_current_context_and_target() -> None:
    first_visit = ObservedVisit(
        visit_id="visit-1",
        diagnoses=(3, 1),
        procedures=(8, 4),
        medications=(11, 5),
        occurred_at=datetime(2020, 1, 1),
    )
    second_visit = ObservedVisit(
        visit_id="visit-2",
        diagnoses=(6,),
        procedures=(),
        medications=(12,),
        occurred_at=datetime(2020, 1, 2),
    )

    example = MedicationSetExample(
        example_id="patient-1:visit-3",
        patient_id="patient-1",
        history=(first_visit, second_visit),
        current_context=_current_context(),
        target_medications=(15, 10),
    )

    assert example.target_visit_id == "visit-3"
    assert example.target_time == datetime(2020, 1, 3)
    assert example.history[0].diagnoses == (3, 1)
    assert example.history[0].medications == (11, 5)
    assert example.target_medications == (15, 10)
    assert not hasattr(example.current_context, "medications")


def test_first_visit_prediction_can_have_empty_history() -> None:
    example = MedicationSetExample(
        example_id="patient-1:visit-1",
        patient_id="patient-1",
        history=(),
        current_context=CurrentVisitContext(
            visit_id="visit-1",
            diagnoses=(1,),
            procedures=(),
        ),
        target_medications=(2,),
    )

    assert example.history == ()


def test_example_is_immutable() -> None:
    example = MedicationSetExample(
        example_id="patient-1:visit-3",
        patient_id="patient-1",
        history=(),
        current_context=_current_context(),
        target_medications=(15,),
    )

    with pytest.raises(FrozenInstanceError):
        example.patient_id = "another-patient"  # type: ignore[misc]


def test_target_visit_cannot_appear_in_history() -> None:
    target_as_history = ObservedVisit(
        visit_id="visit-3",
        diagnoses=(1,),
        procedures=(2,),
        medications=(3,),
    )

    with pytest.raises(ValueError, match="target visit"):
        MedicationSetExample(
            example_id="patient-1:visit-3",
            patient_id="patient-1",
            history=(target_as_history,),
            current_context=_current_context(),
            target_medications=(15,),
        )


def test_target_medication_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        MedicationSetExample(
            example_id="patient-1:visit-3",
            patient_id="patient-1",
            history=(),
            current_context=_current_context(),
            target_medications=(15, 15),
        )


def test_code_collections_must_be_immutable_tuples() -> None:
    with pytest.raises(TypeError, match="diagnoses must be a tuple"):
        CurrentVisitContext(
            visit_id="visit-3",
            diagnoses=[1],  # type: ignore[arg-type]
            procedures=(),
        )
