import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from ehrdrec.models import MRDTR
from ehrdrec.models.utils import create_ehr_adjacency_matrix

from ehrdrec.datasets.mrdtr import (
    MRDTRBatch,
    MRDTRDataset,
    build_mrdtr_graph,
    collate_mrdtr_examples,
)


def _mrdtr_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "patient_id": [10, 10, 20, 20, 20],
            "time": [0, 3, 0, 2, 5],
            "diagnosis_ids": [[1, 2], [3], [2], [4], [5]],
            "procedure_ids": [[7], [8], [7], [9], [10]],
            "atc_ids": [[0, 2], [1], [3], [4], [5]],
        }
    )


def test_build_mrdtr_graph_uses_history_for_edges_and_last_visit_for_label():
    graph = build_mrdtr_graph(_mrdtr_df(), time_col="time")

    assert graph["patient"][0]["diagnosis"] == {1: [0.0], 2: [0.0]}
    assert graph["diagnosis"][1] == {0: [0.0]}
    assert graph["procedure"][7] == {0: [0.0], 1: [0.0]}
    assert graph["medication"][3] == {1: [0.0]}

    assert graph["temporal_feature"] == {0: 3.0, 1: 5.0}
    assert graph["label"] == {0: [1], 1: [5]}


def test_mrdtr_dataset_returns_model_feature_dict_and_dense_label():
    dataset = MRDTRDataset(
        _mrdtr_df(),
        n_medications=6,
        time_col="time",
        hop_num=3,
    )

    features, target = dataset[0]

    assert set(features) == {
        "hop_node_indices",
        "hop_node_temporal_features",
        "central_node_temporal_feature",
        "diagnosis_code_lists",
        "procedure_code_lists",
    }
    assert features["hop_node_indices"][0] == [0]
    assert features["hop_node_indices"][1] == [[1, 2], [7], [0, 2]]
    assert features["central_node_temporal_feature"] == 3.0
    assert features["diagnosis_code_lists"] == [[1, 2], [3]]
    assert target.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_collate_mrdtr_examples_supports_batch_size_one():
    dataset = MRDTRDataset(_mrdtr_df(), n_medications=6, time_col="time")
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_mrdtr_examples)

    features, target = next(iter(loader))

    assert isinstance(features, MRDTRBatch)
    assert features["central_node_temporal_feature"] == 3.0
    assert features.to("cpu") is features
    assert features.size(0) == 1
    assert target.shape == (1, 6)


def test_collate_mrdtr_examples_rejects_larger_batches():
    dataset = MRDTRDataset(_mrdtr_df(), n_medications=6, time_col="time")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_mrdtr_examples)

    with pytest.raises(ValueError, match="batch_size=1"):
        next(iter(loader))


def test_build_mrdtr_graph_keeps_patient_indices_dense_when_skipping_short_histories():
    frame = pl.DataFrame(
        {
            "patient_id": [10, 20, 20],
            "time": [0, 0, 1],
            "diagnosis_ids": [[1], [2], [3]],
            "procedure_ids": [[1], [2], [3]],
            "atc_ids": [[1], [2], [3]],
        }
    )

    graph = build_mrdtr_graph(frame, time_col="time")

    assert list(graph["patient"]) == [0]
    assert graph["label"] == {0: [3]}


def test_build_mrdtr_graph_accepts_string_timestamps():
    frame = pl.DataFrame(
        {
            "patient_id": [10, 10],
            "admission_time": [
                "2020-01-01 00:00:00",
                "2020-01-04 12:00:00",
            ],
            "diagnosis_ids": [[1], [2]],
            "procedure_ids": [[3], [4]],
            "atc_ids": [[0], [1]],
        }
    )

    graph = build_mrdtr_graph(frame)

    assert graph["patient"][0]["diagnosis"] == {1: [0.0]}
    assert graph["temporal_feature"] == {0: 3.5}
    assert graph["label"] == {0: [1]}


def test_create_ehr_adjacency_matrix_accepts_medication_id_lists():
    frame = pl.DataFrame(
        {
            "atc_ids": [[0, 2], [1], [0, 2]],
        }
    )

    adjacency = create_ehr_adjacency_matrix(
        frame,
        medication_col="atc_ids",
        n_medications=3,
    )

    assert adjacency.shape == (3, 3)
    assert adjacency[0, 2].item() == 1.0
    assert adjacency[2, 0].item() == 1.0
    assert adjacency.diag().sum().item() == 0.0


def test_create_ehr_adjacency_matrix_requires_vocab_size_for_sparse_ids():
    frame = pl.DataFrame({"atc_ids": [[0, 3]]})

    with pytest.raises(ValueError, match="maximum medication id"):
        create_ehr_adjacency_matrix(
            frame,
            medication_col="atc_ids",
            n_medications=3,
        )


def test_mrdtr_forward_accepts_single_patient_histories():
    n_medications = 6
    model = MRDTR(
        n_diagnoses=12,
        n_procedures=12,
        n_medications=n_medications,
        n_patients=4,
        ehr_adjacency_matrix=torch.zeros(n_medications, n_medications),
        ddi_adjacency_matrix=torch.zeros(n_medications, n_medications),
        device=torch.device("cpu"),
        hop_num=2,
    )

    output = model(
        hop_node_indices=[[0], [[1, 2], [3], [0, 2]], [0, 1]],
        hop_node_temporal_features=[[3.0], [0.0, 1.0, 0.0], [0.0, 1.0]],
        central_node_temporal_feature=3.0,
        diagnosis_code_lists=[[1, 2], [3]],
        procedure_code_lists=[[3], [4]],
    )

    assert output["predictions"].shape == (1, n_medications)
    assert output["losses"]["ddi_loss"].shape == ()
