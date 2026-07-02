from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset


MRDTRGraph = dict[str, dict]


@dataclass(slots=True)
class MRDTRBatch:
    hop_node_indices: list[Any]
    hop_node_temporal_features: list[list[float]]
    central_node_temporal_feature: float
    diagnosis_code_lists: list[list[int]]
    procedure_code_lists: list[list[int]]

    def __getitem__(self, key: str):
        return getattr(self, key)

    def size(self, dim: int | None = None):
        if dim is None:
            return torch.Size([1])
        if dim == 0:
            return 1
        raise IndexError("MRDTRBatch only has a batch dimension.")

    def to(self, device):
        return self


def build_mrdtr_graph(
    frame: pl.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    time_col: str = "admission_time",
    diagnosis_col: str = "diagnosis_ids",
    procedure_col: str = "procedure_ids",
    medication_col: str = "atc_ids",
    min_visits: int = 2,
) -> MRDTRGraph:
    """
    Build the original MRDTR heterogeneous graph from a SetSequenceProcessor frame.

    The graph uses dense patient indices, matching the original implementation's
    enumerate(data_list) behavior and the MRDTR patient embedding contract.
    """
    graph: MRDTRGraph = {
        "patient": {},
        "diagnosis": {},
        "procedure": {},
        "medication": {},
        "temporal_feature": {},
        "label": {},
    }

    patient_index = 0
    for patient_df in _iter_patient_frames(
        frame,
        patient_id_col=patient_id_col,
        time_col=time_col,
    ):
        if patient_df.height < min_visits:
            continue

        graph["patient"][patient_index] = {
            "diagnosis": {},
            "procedure": {},
            "medication": {},
        }

        rows = list(patient_df.iter_rows(named=True))
        first_time = rows[0][time_col]

        for row in rows[:-1]:
            timestamp = _relative_time(first_time, row[time_col])

            for diagnosis_id in _as_int_list(row[diagnosis_col]):
                _append_edge(
                    graph=graph,
                    patient_index=patient_index,
                    node_type="diagnosis",
                    node_id=diagnosis_id,
                    timestamp=timestamp,
                )

            for procedure_id in _as_int_list(row[procedure_col]):
                _append_edge(
                    graph=graph,
                    patient_index=patient_index,
                    node_type="procedure",
                    node_id=procedure_id,
                    timestamp=timestamp,
                )

            for medication_id in _as_int_list(row[medication_col]):
                _append_edge(
                    graph=graph,
                    patient_index=patient_index,
                    node_type="medication",
                    node_id=medication_id,
                    timestamp=timestamp,
                )

        label_row = rows[-1]
        graph["temporal_feature"][patient_index] = _relative_time(
            first_time,
            label_row[time_col],
        )
        graph["label"][patient_index] = _as_int_list(label_row[medication_col])
        patient_index += 1

    return graph


class MRDTRDataset(Dataset):
    """
    Dataset adapter from SetSequenceProcessor frames to MRDTR inputs.

    The current original MRDTR forward path consumes Python lists for hop nodes,
    so pair this dataset with collate_mrdtr_examples and batch_size=1.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        n_medications: int,
        graph: MRDTRGraph | None = None,
        patient_id_col: str = "patient_id",
        time_col: str = "admission_time",
        diagnosis_col: str = "diagnosis_ids",
        procedure_col: str = "procedure_ids",
        medication_col: str = "atc_ids",
        hop_num: int = 3,
        min_visits: int = 2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.frame = frame
        self.n_medications = n_medications
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.diagnosis_col = diagnosis_col
        self.procedure_col = procedure_col
        self.medication_col = medication_col
        self.hop_num = hop_num
        self.min_visits = min_visits
        self.dtype = dtype
        self.graph = graph or build_mrdtr_graph(
            frame,
            patient_id_col=patient_id_col,
            time_col=time_col,
            diagnosis_col=diagnosis_col,
            procedure_col=procedure_col,
            medication_col=medication_col,
            min_visits=min_visits,
        )
        self.patient_indices = sorted(self.graph["label"])
        self.histories = self._build_patient_histories(frame)

    def __len__(self) -> int:
        return len(self.patient_indices)

    def __getitem__(self, idx: int):
        patient_index = self.patient_indices[idx]
        hop_node_indices, hop_node_temporal_features = self._build_hops(patient_index)
        diagnosis_code_lists, procedure_code_lists = self.histories[patient_index]

        features = {
            "hop_node_indices": hop_node_indices,
            "hop_node_temporal_features": hop_node_temporal_features,
            "central_node_temporal_feature": self.graph["temporal_feature"][
                patient_index
            ],
            "diagnosis_code_lists": diagnosis_code_lists,
            "procedure_code_lists": procedure_code_lists,
        }
        target = self._ids_to_dense(self.graph["label"][patient_index])

        return features, target

    def _build_patient_histories(
        self,
        frame: pl.DataFrame,
    ) -> dict[int, tuple[list[list[int]], list[list[int]]]]:
        histories = {}

        patient_index = 0
        for patient_df in _iter_patient_frames(
            frame,
            patient_id_col=self.patient_id_col,
            time_col=self.time_col,
        ):
            if patient_df.height < self.min_visits:
                continue

            diagnosis_history = []
            procedure_history = []

            for row in patient_df.iter_rows(named=True):
                diagnosis_history.append(_as_int_list(row[self.diagnosis_col]))
                procedure_history.append(_as_int_list(row[self.procedure_col]))

            histories[patient_index] = (diagnosis_history, procedure_history)
            patient_index += 1

        return histories

    def _build_hops(self, patient_index: int) -> tuple[list[Any], list[list[float]]]:
        hop_node_indices: list[Any] = [[patient_index]]
        hop_node_temporal_features: list[list[float]] = [
            [float(self.graph["temporal_feature"][patient_index])]
        ]

        current_patients = {patient_index}
        current_codes: dict[str, set[int]] = {
            "diagnosis": set(),
            "procedure": set(),
            "medication": set(),
        }

        for hop_index in range(1, self.hop_num + 1):
            if hop_index % 2 == 1:
                codes, temporal_features = self._codes_for_patients(current_patients)
                hop_node_indices.append(
                    [
                        sorted(codes["diagnosis"]),
                        sorted(codes["procedure"]),
                        sorted(codes["medication"]),
                    ]
                )
                hop_node_temporal_features.append(temporal_features)
                current_codes = codes
            else:
                patients, temporal_features = self._patients_for_codes(current_codes)
                hop_node_indices.append(sorted(patients))
                hop_node_temporal_features.append(temporal_features)
                current_patients = patients or current_patients

        return hop_node_indices, hop_node_temporal_features

    def _codes_for_patients(
        self,
        patient_indices: set[int],
    ) -> tuple[dict[str, set[int]], list[float]]:
        codes: dict[str, set[int]] = {
            "diagnosis": set(),
            "procedure": set(),
            "medication": set(),
        }
        temporal_features: list[float] = []

        for patient_index in sorted(patient_indices):
            patient_graph = self.graph["patient"].get(patient_index, {})
            for node_type in ("diagnosis", "procedure", "medication"):
                for node_id, timestamps in patient_graph.get(node_type, {}).items():
                    codes[node_type].add(int(node_id))
                    temporal_features.extend(float(value) for value in timestamps)

        return codes, temporal_features

    def _patients_for_codes(
        self,
        codes: dict[str, set[int]],
    ) -> tuple[set[int], list[float]]:
        patient_indices: set[int] = set()
        temporal_features: list[float] = []

        for node_type in ("diagnosis", "procedure", "medication"):
            for node_id in sorted(codes[node_type]):
                patient_edges = self.graph[node_type].get(node_id, {})
                for patient_index, timestamps in patient_edges.items():
                    patient_indices.add(int(patient_index))
                    temporal_features.extend(float(value) for value in timestamps)

        return patient_indices, temporal_features

    def _ids_to_dense(self, ids: list[int]) -> torch.Tensor:
        target = torch.zeros(self.n_medications, dtype=self.dtype)
        if ids:
            target[ids] = 1.0
        return target


def collate_mrdtr_examples(batch):
    if len(batch) != 1:
        raise ValueError(
            "MRDTRDataset currently supports batch_size=1 because the original "
            "MRDTR forward path consumes unbatched Python lists."
        )

    features, target = batch[0]
    return MRDTRBatch(**features), target.unsqueeze(0)


def _append_edge(
    *,
    graph: MRDTRGraph,
    patient_index: int,
    node_type: str,
    node_id: int,
    timestamp: float,
) -> None:
    graph[node_type].setdefault(node_id, {}).setdefault(patient_index, []).append(
        timestamp
    )
    graph["patient"][patient_index][node_type].setdefault(node_id, []).append(timestamp)


def _iter_patient_frames(
    frame: pl.DataFrame,
    *,
    patient_id_col: str,
    time_col: str,
):
    sorted_frame = frame.sort([patient_id_col, time_col])
    for _, patient_df in sorted_frame.group_by(patient_id_col, maintain_order=True):
        yield patient_df


def _as_int_list(values) -> list[int]:
    if values is None:
        return []

    if hasattr(values, "to_list"):
        values = values.to_list()

    return [int(value) for value in values]


def _relative_time(start: Hashable, value: Hashable) -> float:
    start = _coerce_time_value(start)
    value = _coerce_time_value(value)
    delta = value - start

    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds() / 86_400)

    if hasattr(delta, "days"):
        return float(delta.days)

    return float(delta)


def _coerce_time_value(value: Hashable):
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return value
