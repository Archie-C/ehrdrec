from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset

from ehrdrec.utils import ReservedId


class HypeMedDataset(Dataset):
    """Dataset adapter for HypeMed.

    Each sample is a patient sequence or a target-visit history window. The
    collate function pads code lists and flattens valid visit targets to match
    HypeMed's visit-level logits.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        n_diagnoses: int,
        n_procedures: int,
        n_medications: int,
        patient_id_col: str = "patient_id",
        time_col: str = "admission_time",
        diagnosis_col: str = "diagnosis_ids",
        procedure_col: str = "procedure_ids",
        medication_col: str = "atc_ids",
        medication_is_multihot: bool = False,
        edge_id_col: str | None = None,
        min_visits: int = 1,
        sample_all_visits: bool = True,
        look_back: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.frame = frame.sort([patient_id_col, time_col])
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.n_medications = n_medications
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.diagnosis_col = diagnosis_col
        self.procedure_col = procedure_col
        self.medication_col = medication_col
        self.medication_is_multihot = medication_is_multihot
        self.edge_id_col = edge_id_col
        self.dtype = dtype
        self.samples = []
        self.patient_frames = []

        for _, patient_df in self.frame.group_by(patient_id_col, maintain_order=True):
            if patient_df.height < min_visits:
                continue

            if sample_all_visits:
                for visit_idx in range(min_visits - 1, patient_df.height):
                    start_idx = 0 if look_back is None else max(0, visit_idx - look_back + 1)
                    self.samples.append({
                        "patient_df": patient_df,
                        "start_idx": start_idx,
                        "end_idx": visit_idx + 1,
                    })
            else:
                self.patient_frames.append(patient_df)

    def __len__(self) -> int:
        return len(self.samples) if self.samples else len(self.patient_frames)

    def __getitem__(self, idx: int):
        if self.samples:
            sample = self.samples[idx]
            patient_df = sample["patient_df"].slice(
                sample["start_idx"],
                sample["end_idx"] - sample["start_idx"],
            )
        else:
            patient_df = self.patient_frames[idx]

        diagnoses = []
        procedures = []
        medications = []
        targets = []
        visit2edge = []

        for row in patient_df.iter_rows(named=True):
            diagnosis_ids = self._code_ids(row[self.diagnosis_col], self.n_diagnoses)
            procedure_ids = self._code_ids(row[self.procedure_col], self.n_procedures)
            medication_ids = self._medication_ids(row[self.medication_col])

            diagnoses.append(diagnosis_ids)
            procedures.append(procedure_ids)
            medications.append(self._code_ids(medication_ids, self.n_medications))
            targets.append(self._ids_to_dense(medication_ids))
            visit2edge.append(int(row[self.edge_id_col]) if self.edge_id_col else 0)

        return {
            "diagnoses": diagnoses,
            "procedures": procedures,
            "medications": medications,
            "visit2edge": visit2edge,
            "seq_length": len(diagnoses),
        }, torch.stack(targets)

    def _code_ids(self, values: Any, vocab_size: int) -> list[int]:
        ids = _as_int_list(values)
        if not ids:
            ids = [int(ReservedId.UNK)]
        return [id_ for id_ in ids if 0 <= id_ < vocab_size]

    def _medication_ids(self, values: Any) -> list[int]:
        if self.medication_is_multihot:
            values = _as_python_list(values)
            return [idx for idx, value in enumerate(values) if value > 0]
        return _as_int_list(values)

    def _ids_to_dense(self, ids: list[int]) -> torch.Tensor:
        target = torch.zeros(self.n_medications, dtype=self.dtype)
        valid_ids = [id_ for id_ in ids if 0 <= id_ < self.n_medications]
        if valid_ids:
            target[valid_ids] = 1.0
        return target


def collate_hypemed_examples(batch):
    features, targets = zip(*batch)
    batch_size = len(features)
    max_seq_length = max(feature["seq_length"] for feature in features)
    max_diag_num = _max_code_count(features, "diagnoses")
    max_proc_num = _max_code_count(features, "procedures")
    max_med_num = _max_code_count(features, "medications")
    n_medications = targets[0].shape[-1]
    target_dtype = targets[0].dtype

    diagnoses = torch.full((batch_size, max_seq_length, max_diag_num), int(ReservedId.PAD), dtype=torch.long)
    procedures = torch.full((batch_size, max_seq_length, max_proc_num), int(ReservedId.PAD), dtype=torch.long)
    medications = torch.full((batch_size, max_seq_length, max_med_num), int(ReservedId.PAD), dtype=torch.long)

    total_visits = batch_size * max_seq_length
    true_visit_idx = torch.zeros(total_visits, dtype=torch.bool)
    attn_mask = torch.ones((total_visits, total_visits), dtype=torch.bool)
    visit2edge_idx = []
    flat_targets = []

    for batch_idx, (feature, target) in enumerate(zip(features, targets)):
        seq_len = feature["seq_length"]
        offset = batch_idx * max_seq_length
        true_visit_idx[offset: offset + seq_len] = True

        for target_visit in range(seq_len):
            row_idx = offset + target_visit
            allowed = offset + torch.arange(target_visit + 1)
            attn_mask[row_idx, allowed] = False

        _copy_code_lists(diagnoses[batch_idx], feature["diagnoses"])
        _copy_code_lists(procedures[batch_idx], feature["procedures"])
        _copy_code_lists(medications[batch_idx], feature["medications"])
        visit2edge_idx.extend(feature["visit2edge"])
        flat_targets.append(target[:seq_len])

    return {
        "diagnoses": diagnoses,
        "procedures": procedures,
        "medications": medications,
        "attn_mask": attn_mask,
        "true_visit_idx": true_visit_idx,
        "visit2edge_idx": torch.tensor(visit2edge_idx, dtype=torch.long),
    }, torch.cat(flat_targets, dim=0).reshape(-1, n_medications).to(target_dtype)


def _copy_code_lists(output: torch.Tensor, code_lists: list[list[int]]) -> None:
    for visit_idx, code_ids in enumerate(code_lists):
        output[visit_idx, :len(code_ids)] = torch.tensor(code_ids, dtype=torch.long)


def _max_code_count(features, key: str) -> int:
    return max(max(len(code_ids) for code_ids in feature[key]) for feature in features)


def _as_python_list(values: Any) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _as_int_list(values: Any) -> list[int]:
    return [int(value) for value in _as_python_list(values)]
