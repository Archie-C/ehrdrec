import ast
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from ehrdrec.datasets import SHAPEDataset, collate_shape_examples
from ehrdrec.utils import ReservedId


def _shape_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "patient_id": [1, 1, 2, 1],
            "time": [2, 0, 0, 1],
            "diagnosis_ids": [[4], [2, 3], [], [5]],
            "procedure_ids": [[7, 8], [6], [9], []],
            "atc_ids": [[3], [0, 2], [4], []],
        }
    )


def test_shape_dataset_returns_one_sorted_patient_sequence():
    dataset = SHAPEDataset(
        _shape_df(),
        n_diagnoses=12,
        n_procedures=12,
        n_medications=6,
        time_col="time",
    )

    features, target = dataset[0]

    assert features["seq_length"] == 3
    assert features["diseases"] == [[2, 3], [5], [4]]
    assert features["procedures"] == [[6], [int(ReservedId.UNK)], [7, 8]]
    assert features["medications"] == [[0, 2], [int(ReservedId.UNK)], [3]]
    assert target.shape == (3, 6)
    assert target[0].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert target[1].sum().item() == 0.0


def test_collate_shape_examples_pads_codes_visits_and_masks():
    dataset = SHAPEDataset(
        _shape_df(),
        n_diagnoses=12,
        n_procedures=12,
        n_medications=6,
        time_col="time",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_shape_examples,
    )

    features, target = next(iter(loader))

    assert features["diseases"].shape == (2, 3, 2)
    assert features["procedures"].shape == (2, 3, 2)
    assert features["medications"].shape == (2, 3, 2)
    assert features["d_mask_matrix"].shape == (2, 3, 2)
    assert features["seq_length"].tolist() == [3, 1]
    assert target.shape == (2, 6)

    assert features["diseases"][0, 1].tolist() == [5, int(ReservedId.PAD)]
    assert features["d_mask_matrix"][0, 1].tolist() == [0.0, -1e9]
    assert features["diseases"][1, 1:].eq(int(ReservedId.PAD)).all()
    assert features["d_mask_matrix"][1, 1:].eq(-1e9).all()
    assert target[1].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_shape_dataset_accepts_multihot_medication_column():
    frame = pl.DataFrame(
        {
            "patient_id": [1, 1],
            "time": [0, 1],
            "diagnosis_ids": [[2], [3]],
            "procedure_ids": [[4], [5]],
            "medication_multihot": [
                [1, 0, 1, 0],
                [0, 1, 0, 0],
            ],
        }
    )
    dataset = SHAPEDataset(
        frame,
        n_diagnoses=8,
        n_procedures=8,
        n_medications=4,
        time_col="time",
        medication_col="medication_multihot",
        medication_is_multihot=True,
    )

    features, target = dataset[0]

    assert features["medications"] == [[0, 2], [1]]
    assert target.tolist() == [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]


def test_shape_forward_accepts_feature_dict_without_adapter():
    source = ast.parse(Path("src/ehrdrec/models/torch/original/shape.py").read_text())
    shape_class = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "SHAPE"
    )
    forward = next(
        node
        for node in shape_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )

    assert [arg.arg for arg in forward.args.args] == ["self", "features"]

    constants = {
        node.value
        for node in ast.walk(forward)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "diseases",
        "procedures",
        "medications",
        "d_mask_matrix",
        "p_mask_matrix",
        "m_mask_matrix",
        "seq_length",
        "predictions",
        "losses",
        "ddi_loss",
    }.issubset(constants)

