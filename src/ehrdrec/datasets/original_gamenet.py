import polars as pl
import torch
from torch.utils.data import Dataset


class OriginalGAMENetDataset(Dataset):
    """
    Dataset adapter for the upstream/original GAMENet implementation.

    Each item returns a Python list of visits:
        [[diagnosis_ids, procedure_ids, medication_ids], ...]
    plus the current visit target as a dense multi-hot tensor.
    """

    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        target_col: str,
        patient_id_col: str,
        time_col: str,
        diagnosis_col: str = "diagnosis_ids",
        procedure_col: str = "procedure_ids",
        medication_history_col: str | None = None,
        look_back: int | None = None,
        medication_history_is_multihot: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.target_col = target_col
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.diagnosis_col = diagnosis_col
        self.procedure_col = procedure_col
        self.medication_history_col = medication_history_col or target_col
        self.look_back = look_back
        self.medication_history_is_multihot = medication_history_is_multihot
        self.dtype = dtype

        self.data_frame = multi_hot_data_frame.sort([patient_id_col, time_col])
        self.samples = []

        for _, patient_df in self.data_frame.group_by(patient_id_col):
            n_visits = patient_df.height

            for visit_idx in range(n_visits):
                if look_back is None:
                    start_idx = 0
                else:
                    start_idx = max(0, visit_idx - look_back)

                self.samples.append(
                    {
                        "patient_df": patient_df,
                        "start_idx": start_idx,
                        "end_idx": visit_idx + 1,
                        "target_idx": visit_idx,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        patient_df = sample["patient_df"]
        history_df = patient_df.slice(
            sample["start_idx"],
            sample["end_idx"] - sample["start_idx"],
        )
        target_row = patient_df.row(sample["target_idx"], named=True)

        history = []
        for row in history_df.iter_rows(named=True):
            history.append(
                [
                    self._to_int_list(row[self.diagnosis_col]),
                    self._to_int_list(row[self.procedure_col]),
                    self._medications_to_ids(row[self.medication_history_col]),
                ]
            )

        target = torch.tensor(target_row[self.target_col], dtype=self.dtype)

        return history, target

    def _to_int_list(self, values) -> list[int]:
        if values is None:
            return []
        return [int(value) for value in values]

    def _medications_to_ids(self, values) -> list[int]:
        if values is None:
            return []

        if self.medication_history_is_multihot:
            return [idx for idx, value in enumerate(values) if value > 0]

        return self._to_int_list(values)


def collate_original_gamenet(batch):
    histories, targets = zip(*batch)
    return list(histories), torch.stack(targets)
