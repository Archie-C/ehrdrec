import polars as pl
import torch
from torch.utils.data import Dataset


class MultiHotDataset(Dataset):
    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        target_col: str,
        n_diagnoses: int,
        n_procedures: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.data_frame = multi_hot_data_frame
        self.target_col = target_col
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.dtype = dtype

    def __len__(self) -> int:
        return self.data_frame.height

    def __getitem__(self, idx: int):
        row = self.data_frame.row(idx, named=True)

        x = torch.cat([
            self._ids_to_dense(row["diagnosis_ids"], self.n_diagnoses),
            self._ids_to_dense(row["procedure_ids"], self.n_procedures),
        ])
        y = torch.tensor(row[self.target_col], dtype=self.dtype)

        return x, y

    def _ids_to_dense(self, ids: list[int], size: int) -> torch.Tensor:
        vec = torch.zeros(size, dtype=self.dtype)
        if ids:
            vec[ids] = 1.0
        return vec
    
class MultiHotDatasetWithPatientLookBack(Dataset):
    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        target_col: str,
        n_diagnoses: int,
        n_procedures: int,
        patient_id_col: str,
        time_col: str,
        look_back: int = 3,
        dtype: torch.dtype = torch.float32,
    ):
        self.target_col = target_col
        self.n_diagnoses = n_diagnoses
        self.n_procedures = n_procedures
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.look_back = look_back
        self.dtype = dtype

        self.data_frame = multi_hot_data_frame.sort(
            [patient_id_col, time_col]
        )

        self.samples = []

        for patient_id, patient_df in self.data_frame.group_by(patient_id_col):
            n_visits = patient_df.height

            for visit_idx in range(n_visits):
                start_idx = max(0, visit_idx - look_back)

                self.samples.append(
                    {
                        "patient_df": patient_df,
                        "start_idx": start_idx,
                        "end_idx": visit_idx + 1,
                        "target_idx": visit_idx,
                    }
                )
                
    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        patient_df = sample["patient_df"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        target_idx = sample["target_idx"]

        history_df = patient_df.slice(start_idx, end_idx - start_idx)
        target_row = patient_df.row(target_idx, named=True)
        n_visits = end_idx - start_idx

        diag = torch.zeros(n_visits, self.n_diagnoses, dtype=self.dtype)
        proc = torch.zeros(n_visits, self.n_procedures, dtype=self.dtype)
        for i, ids_row in enumerate(history_df["diagnosis_ids"].to_list()):
            if ids_row:
                diag[i, ids_row] = 1.0
        for i, ids_row in enumerate(history_df["procedure_ids"].to_list()):
            if ids_row:
                proc[i, ids_row] = 1.0

        x = {
            "diagnoses": diag,
            "procedures": proc,
            "medication_history": torch.tensor(
                history_df[self.target_col].to_list(),
                dtype=self.dtype,
            ),
        }

        y = torch.tensor(
            target_row[self.target_col],
            dtype=self.dtype,
        )

        return x, y
    
    def __len__(self):
        return len(self.samples)
    
class MultiHotDatasetWithAllATCLevels(Dataset):
    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        self.data_frame = multi_hot_data_frame
        self.dtype = dtype

    def __len__(self) -> int:
        return self.data_frame.height

    def __getitem__(self, idx: int):
        row = self.data_frame.row(idx, named=True)
        
        x = {
            "diagnoses": torch.tensor(
                row["diagnosis_multihot"],
                dtype=self.dtype,
            ),
            "procedures": torch.tensor(
                row["procedure_multihot"],
                dtype=self.dtype,
            ),
        }
        
        y = {
            "atc5": torch.tensor(
                row["atc5_multihot"],
                dtype=self.dtype,
            ),
            "atc4": torch.tensor(
                row["atc4_multihot"],
                dtype=self.dtype,
            ),
            "atc3": torch.tensor(
                row["atc3_multihot"],
                dtype=self.dtype,
            ),
            "atc2": torch.tensor(
                row["atc2_multihot"],
                dtype=self.dtype,
            ),
            "atc1": torch.tensor(
                row["atc1_multihot"],
                dtype=self.dtype,
            ),
        }
        return x, y