from datetime import datetime

import pytest
import torch

from ehrdrec.models.adapter import VocabularySizes
from ehrdrec.models.cognet.adapter import COGNetAdapter
from ehrdrec.models.dmnc.adapter import DMNCAdapter
from ehrdrec.models.gamenet.adapter import GAMENetAdapter
from ehrdrec.models.micron.adapter import MICRONAdapter
from ehrdrec.models.molerec.adapter import MoleRecAdapter
from ehrdrec.models.retain.adapter import RETAINAdapter
from ehrdrec.models.safedrug.adapter import SafeDrugAdapter
from ehrdrec.tasks.medication_set import (
    CurrentVisitContext,
    MedicationSetExample,
    ObservedVisit,
)


VOCABULARY_SIZES = VocabularySizes(
    diagnoses=20,
    procedures=10,
    medications=16,
)


def example() -> MedicationSetExample:
    return MedicationSetExample(
        example_id="patient-1:visit-3",
        patient_id="patient-1",
        history=(
            ObservedVisit(
                visit_id="visit-1",
                diagnoses=(3, 1),
                procedures=(8, 4),
                medications=(11, 5),
                occurred_at=datetime(2020, 1, 1),
            ),
            ObservedVisit(
                visit_id="visit-2",
                diagnoses=(6,),
                procedures=(2,),
                medications=(12,),
                occurred_at=datetime(2020, 1, 2),
            ),
        ),
        current_context=CurrentVisitContext(
            visit_id="visit-3",
            diagnoses=(9, 2),
            procedures=(7,),
            occurred_at=datetime(2020, 1, 4),
        ),
        target_medications=(15, 10),
    )


def test_gamenet_keeps_only_historical_medications_in_input() -> None:
    batch = GAMENetAdapter(VOCABULARY_SIZES).training_collate([example()])

    assert batch["x"][0]["medications"] == [11, 5]
    assert batch["x"][-1]["medications"] == []
    assert torch.equal(
        batch["Y"].nonzero(),
        torch.tensor([[0, 10], [0, 15]]),
    )


@pytest.mark.parametrize("adapter_type", [MICRONAdapter, SafeDrugAdapter])
def test_clinical_adapters_exclude_medications(adapter_type: type) -> None:
    model_input = adapter_type(VOCABULARY_SIZES).adapt_input(
        example(),
        teacher_forcing=False,
    )

    assert all("medications" not in visit for visit in model_input)


def test_molerec_and_dmnc_support_batches_without_changing_order() -> None:
    task_example = example()
    mole_batch = MoleRecAdapter(VOCABULARY_SIZES).training_collate(
        [task_example, task_example]
    )
    dmnc_input = DMNCAdapter(VOCABULARY_SIZES).adapt_input(
        task_example,
        teacher_forcing=False,
    )

    assert len(mole_batch["x"]) == 2
    assert mole_batch["Y"].shape == (2, 16)
    assert dmnc_input[0] == {
        "diagnoses": [3, 1],
        "procedures": [8, 4],
    }


def test_retain_combines_namespaces_and_computes_elapsed_days() -> None:
    adapter = RETAINAdapter(VOCABULARY_SIZES, use_time=True)

    model_input = adapter.adapt_input(example(), teacher_forcing=False)

    assert adapter.input_vocab_size == 30
    assert model_input[0] == {"codes": [3, 1, 28, 24], "time": 0.0}
    assert model_input[1]["time"] == 1.0
    assert model_input[2]["time"] == 2.0


def test_cognet_hides_current_target_outside_teacher_forcing() -> None:
    adapter = COGNetAdapter(
        VOCABULARY_SIZES,
        medication_frequencies={5: 20, 10: 7, 11: 2, 12: 4, 15: 1},
    )

    training_input = adapter.adapt_input(example(), teacher_forcing=True)
    inference_input = adapter.adapt_input(example(), teacher_forcing=False)

    assert training_input[0]["medications"] == [11, 5]
    assert training_input[-1]["medications"] == [15, 10]
    assert inference_input[-1]["medications"] == []


def test_single_history_adapters_reject_larger_batches() -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        GAMENetAdapter(VOCABULARY_SIZES).training_collate(
            [example(), example()]
        )
