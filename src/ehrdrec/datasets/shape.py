from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset

from ehrdrec.utils import ReservedId


class SHAPEDataset(Dataset):
    """
    Dataset adapter for the original SHAPE implementation.

    By default, each item is one full patient sequence. With
    sample_all_visits=True, each item is one eligible target visit plus its
    preceding history, which is useful for visit-level model comparisons. Use
    collate_shape_examples to pad patient sequences and per-visit code lists
    into the batched tensors expected by SHAPE.forward.
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
        min_visits: int = 1,
        sample_all_visits: bool = False,
        look_back: int | None = None,
        mask_value: float = -1e9,
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
        self.min_visits = min_visits
        self.sample_all_visits = sample_all_visits
        self.look_back = look_back
        self.mask_value = mask_value
        self.dtype = dtype

        self.patient_frames = []
        self.samples = []
        for _, patient_df in self.frame.group_by(
            patient_id_col,
            maintain_order=True,
        ):
            if patient_df.height < min_visits:
                continue

            if sample_all_visits:
                for visit_idx in range(min_visits - 1, patient_df.height):
                    start_idx = 0 if look_back is None else max(0, visit_idx - look_back + 1)
                    self.samples.append(
                        {
                            "patient_df": patient_df,
                            "start_idx": start_idx,
                            "end_idx": visit_idx + 1,
                        }
                    )
            else:
                self.patient_frames.append(patient_df)

    def __len__(self) -> int:
        return len(self.samples) if self.sample_all_visits else len(self.patient_frames)

    def __getitem__(self, idx: int):
        if self.sample_all_visits:
            sample = self.samples[idx]
            patient_df = sample["patient_df"].slice(
                sample["start_idx"],
                sample["end_idx"] - sample["start_idx"],
            )
        else:
            patient_df = self.patient_frames[idx]

        diagnosis_lists: list[list[int]] = []
        procedure_lists: list[list[int]] = []
        medication_lists: list[list[int]] = []
        targets: list[torch.Tensor] = []

        for row in patient_df.iter_rows(named=True):
            diagnosis_lists.append(
                self._code_ids(row[self.diagnosis_col], self.n_diagnoses)
            )
            procedure_lists.append(
                self._code_ids(row[self.procedure_col], self.n_procedures)
            )
            medication_ids = self._medication_ids(row[self.medication_col])
            medication_lists.append(self._code_ids(medication_ids, self.n_medications))
            targets.append(self._ids_to_dense(medication_ids))

        features = {
            "diseases": diagnosis_lists,
            "procedures": procedure_lists,
            "medications": medication_lists,
            "seq_length": len(diagnosis_lists),
            "mask_value": self.mask_value,
        }

        return features, torch.stack(targets)

    def _code_ids(self, values: Any, vocab_size: int) -> list[int]:
        ids = _as_int_list(values)
        if not ids:
            ids = [int(ReservedId.UNK)]

        max_embedding_id = vocab_size + 2
        invalid_ids = [id_ for id_ in ids if id_ < 0 or id_ > max_embedding_id]
        if invalid_ids:
            raise ValueError(
                f"Code ids {invalid_ids} are outside the embedding range "
                f"[0, {max_embedding_id}]."
            )

        return ids

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


def collate_shape_examples(batch):
    features, targets = zip(*batch)

    batch_size = len(features)
    max_seq_length = max(feature["seq_length"] for feature in features)
    max_diag_num = _max_code_count(features, "diseases")
    max_proc_num = _max_code_count(features, "procedures")
    max_med_num = _max_code_count(features, "medications")
    n_medications = targets[0].shape[-1]
    target_dtype = targets[0].dtype
    mask_value = float(features[0]["mask_value"])

    diseases = torch.full(
        (batch_size, max_seq_length, max_diag_num),
        int(ReservedId.PAD),
        dtype=torch.long,
    )
    procedures = torch.full(
        (batch_size, max_seq_length, max_proc_num),
        int(ReservedId.PAD),
        dtype=torch.long,
    )
    medications = torch.full(
        (batch_size, max_seq_length, max_med_num),
        int(ReservedId.PAD),
        dtype=torch.long,
    )
    d_mask_matrix = torch.full(
        (batch_size, max_seq_length, max_diag_num),
        mask_value,
        dtype=target_dtype,
    )
    p_mask_matrix = torch.full(
        (batch_size, max_seq_length, max_proc_num),
        mask_value,
        dtype=target_dtype,
    )
    m_mask_matrix = torch.full(
        (batch_size, max_seq_length, max_med_num),
        mask_value,
        dtype=target_dtype,
    )
    batched_targets = torch.zeros(
        batch_size,
        n_medications,
        dtype=target_dtype,
    )
    seq_length = torch.empty(batch_size, dtype=torch.long)

    for batch_idx, (feature, target) in enumerate(zip(features, targets)):
        seq_len = feature["seq_length"]
        seq_length[batch_idx] = seq_len
        batched_targets[batch_idx] = target[seq_len - 1]

        _copy_code_lists(
            diseases[batch_idx],
            d_mask_matrix[batch_idx],
            feature["diseases"],
        )
        _copy_code_lists(
            procedures[batch_idx],
            p_mask_matrix[batch_idx],
            feature["procedures"],
        )
        _copy_code_lists(
            medications[batch_idx],
            m_mask_matrix[batch_idx],
            feature["medications"],
        )

    return {
        "diseases": diseases,
        "procedures": procedures,
        "medications": medications,
        "d_mask_matrix": d_mask_matrix,
        "p_mask_matrix": p_mask_matrix,
        "m_mask_matrix": m_mask_matrix,
        "seq_length": seq_length,
    }, batched_targets


def _copy_code_lists(
    output: torch.Tensor,
    mask: torch.Tensor,
    code_lists: list[list[int]],
) -> None:
    for visit_idx, code_ids in enumerate(code_lists):
        n_codes = len(code_ids)
        output[visit_idx, :n_codes] = torch.tensor(code_ids, dtype=torch.long)
        mask[visit_idx, :n_codes] = 0.0


def _max_code_count(features, key: str) -> int:
    return max(
        max(len(code_ids) for code_ids in feature[key])
        for feature in features
    )


def _as_python_list(values: Any) -> list:
    if values is None:
        return []

    if hasattr(values, "to_list"):
        return values.to_list()

    return list(values)


def _as_int_list(values: Any) -> list[int]:
    return [int(value) for value in _as_python_list(values)]
