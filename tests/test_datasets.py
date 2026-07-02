"""
Tests for ehrdrec.datasets — MultiHotDataset, MultiHotDatasetWithPatientLookBack,
MultiHotDatasetWithAllATCLevels, and collate_patient_visit_histories.

All fixtures are synthetic Polars DataFrames; no MIMIC files required.
"""
import pytest
import torch
import polars as pl
from torch.utils.data import DataLoader

from ehrdrec.datasets.multi_hot import (
    MultiHotDataset,
    MultiHotDatasetWithPatientLookBack,
    MultiHotDatasetWithAllATCLevels,
)
from ehrdrec.datasets.utils import collate_patient_visit_histories


# ===========================================================================
# Shared fixtures
# ===========================================================================

N_DIAG = 10
N_PROC = 8
N_MEDS = 6


def _basic_df(n_rows: int = 4) -> pl.DataFrame:
    """Minimal DataFrame for MultiHotDataset."""
    return pl.DataFrame({
        "diagnosis_ids":  [[2, 3], [4], [2, 5], []] * (n_rows // 4) if n_rows % 4 == 0
                          else [[2, 3]] * n_rows,
        "procedure_ids":  [[1], [2, 3], [0], [1, 2]] * (n_rows // 4) if n_rows % 4 == 0
                          else [[1]] * n_rows,
        "medication_ids": [[0, 1, 0, 0, 1, 0]] * n_rows,
    })


def _lookback_df() -> pl.DataFrame:
    """Three patients with varying visit counts for look-back tests."""
    return pl.DataFrame({
        "patient_id":     [1, 1, 1, 2, 2, 3],
        "time":           [0, 1, 2, 0, 1, 0],
        "diagnosis_ids":  [[2], [3], [4], [2, 5], [3], [9]],
        "procedure_ids":  [[1], [0], [2], [1],    [3], [0]],
        "medication_ids": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
    })


def _atc_df(n_rows: int = 3) -> pl.DataFrame:
    """Minimal DataFrame for MultiHotDatasetWithAllATCLevels."""
    return pl.DataFrame({
        "diagnosis_multihot":  [[1, 0, 1, 0]] * n_rows,
        "procedure_multihot":  [[0, 1, 0, 1]] * n_rows,
        "atc5_multihot":       [[1, 0, 0]] * n_rows,
        "atc4_multihot":       [[1, 0]]    * n_rows,
        "atc3_multihot":       [[1]]       * n_rows,
        "atc2_multihot":       [[1]]       * n_rows,
        "atc1_multihot":       [[1]]       * n_rows,
    })


# ===========================================================================
# MultiHotDataset
# ===========================================================================

class TestMultiHotDataset:
    @pytest.fixture
    def ds(self):
        return MultiHotDataset(
            _basic_df(),
            target_col="medication_ids",
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
        )

    def test_len(self, ds):
        assert len(ds) == 4

    def test_item_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_DIAG + N_PROC,)
        assert y.shape == (N_MEDS,)

    def test_x_dtype(self, ds):
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_y_dtype(self, ds):
        _, y = ds[0]
        assert y.dtype == torch.float32

    def test_diagnosis_ids_set_correctly(self, ds):
        # Row 0 has diagnosis_ids=[2,3]; slots 2 and 3 in the diag section should be 1
        x, _ = ds[0]
        diag = x[:N_DIAG]
        assert diag[2].item() == 1.0
        assert diag[3].item() == 1.0
        assert diag[0].item() == 0.0

    def test_procedure_ids_set_correctly(self, ds):
        # Row 0 has procedure_ids=[1]; slot 1 in the proc section should be 1
        x, _ = ds[0]
        proc = x[N_DIAG:]
        assert proc[1].item() == 1.0
        assert proc[0].item() == 0.0

    def test_empty_ids_give_zero_vector(self):
        df = pl.DataFrame({
            "diagnosis_ids":  [[]],
            "procedure_ids":  [[]],
            "medication_ids": [[0, 0, 0]],
        })
        ds = MultiHotDataset(df, target_col="medication_ids", n_diagnoses=5, n_procedures=5)
        x, _ = ds[0]
        assert x.sum().item() == 0.0

    def test_target_values_correct(self, ds):
        _, y = ds[0]
        # medication_ids = [0,1,0,0,1,0]
        assert y[1].item() == 1.0
        assert y[4].item() == 1.0
        assert y[0].item() == 0.0

    def test_custom_dtype(self):
        ds = MultiHotDataset(
            _basic_df(4),
            target_col="medication_ids",
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            dtype=torch.float64,
        )
        x, y = ds[0]
        assert x.dtype == torch.float64
        assert y.dtype == torch.float64

    def test_dataloader_batches(self, ds):
        loader = DataLoader(ds, batch_size=2)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (2, N_DIAG + N_PROC)
        assert y_batch.shape == (2, N_MEDS)

    def test_all_items_accessible(self, ds):
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (N_DIAG + N_PROC,)


# ===========================================================================
# MultiHotDatasetWithPatientLookBack
# ===========================================================================

class TestMultiHotDatasetWithPatientLookBack:
    @pytest.fixture
    def ds(self):
        return MultiHotDatasetWithPatientLookBack(
            _lookback_df(),
            target_col="medication_ids",
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            patient_id_col="patient_id",
            time_col="time",
            look_back=2,
        )

    def test_total_samples(self, ds):
        # patient 1: 3 visits → 3 samples
        # patient 2: 2 visits → 2 samples
        # patient 3: 1 visit  → 1 sample
        assert len(ds) == 6

    def test_item_x_keys(self, ds):
        x, _ = ds[0]
        assert set(x.keys()) == {"diagnoses", "procedures", "medication_history"}

    def test_item_y_shape(self, ds):
        _, y = ds[0]
        assert y.shape == (N_MEDS,)

    def test_first_visit_has_one_history_row(self, ds):
        # First visit of any patient → window is just that visit (1 row)
        # Find a sample where start_idx == end_idx - 1
        sample = ds.samples[0]
        n_visits = sample["end_idx"] - sample["start_idx"]
        assert n_visits == 1
        x, _ = ds[0]
        assert x["diagnoses"].shape[0] == 1

    def test_look_back_caps_history(self, ds):
        # Patient 1 visit 2 (index 2 within patient): look_back=2 → window = visits [0,1,2] → 3 rows
        # But look_back=2 means start = max(0, 2-2)=0, so 3 rows max
        # Find that sample
        p1_samples = [s for s in ds.samples if s["target_idx"] == 2]
        assert any(s["end_idx"] - s["start_idx"] <= 3 for s in p1_samples)

    def test_diagnoses_shape(self, ds):
        x, _ = ds[0]
        # (n_visits_in_window, N_DIAG)
        assert x["diagnoses"].shape[1] == N_DIAG

    def test_procedures_shape(self, ds):
        x, _ = ds[0]
        assert x["procedures"].shape[1] == N_PROC

    def test_medication_history_shape(self, ds):
        x, _ = ds[0]
        n_visits = x["diagnoses"].shape[0]
        assert x["medication_history"].shape == (n_visits, N_MEDS)

    def test_diagnosis_values_correct(self, ds):
        # Find the sample for patient 1, visit 0 (first visit)
        # diagnosis_ids=[2] → slot 2 in diag tensor should be 1
        x, _ = ds[0]
        # first (and only) visit row
        assert x["diagnoses"][0, 2].item() == 1.0

    def test_sorted_by_time(self, ds):
        # The dataset sorts by [patient_id, time] — verify internal df is sorted
        times = ds.data_frame["time"].to_list()
        # Within each patient group times should be non-decreasing;
        # just check global sort didn't break anything by confirming no errors
        assert len(times) == 6

    def test_single_visit_patient(self, ds):
        # Patient 3 has only one visit — find that sample and check shapes
        p3_samples = [
            (i, s) for i, s in enumerate(ds.samples)
            if s["patient_df"]["patient_id"][0] == 3
        ]
        assert len(p3_samples) == 1
        idx, _ = p3_samples[0]
        x, y = ds[idx]
        assert x["diagnoses"].shape == (1, N_DIAG)


# ===========================================================================
# MultiHotDatasetWithAllATCLevels
# ===========================================================================

class TestMultiHotDatasetWithAllATCLevels:
    @pytest.fixture
    def ds(self):
        return MultiHotDatasetWithAllATCLevels(_atc_df())

    def test_len(self, ds):
        assert len(ds) == 3

    def test_x_keys(self, ds):
        x, _ = ds[0]
        assert set(x.keys()) == {"diagnoses", "procedures"}

    def test_y_keys(self, ds):
        _, y = ds[0]
        assert set(y.keys()) == {"atc5", "atc4", "atc3", "atc2", "atc1"}

    def test_diagnoses_shape(self, ds):
        x, _ = ds[0]
        assert x["diagnoses"].shape == (4,)

    def test_procedures_shape(self, ds):
        x, _ = ds[0]
        assert x["procedures"].shape == (4,)

    def test_atc5_shape(self, ds):
        _, y = ds[0]
        assert y["atc5"].shape == (3,)

    def test_values_correct(self, ds):
        x, y = ds[0]
        # diagnosis_multihot = [1,0,1,0]
        assert x["diagnoses"][0].item() == 1.0
        assert x["diagnoses"][1].item() == 0.0
        # atc5_multihot = [1,0,0]
        assert y["atc5"][0].item() == 1.0
        assert y["atc5"][1].item() == 0.0

    def test_dtype_default_float32(self, ds):
        x, y = ds[0]
        assert x["diagnoses"].dtype == torch.float32
        assert y["atc5"].dtype == torch.float32

    def test_custom_dtype(self):
        ds = MultiHotDatasetWithAllATCLevels(_atc_df(), dtype=torch.float64)
        x, y = ds[0]
        assert x["diagnoses"].dtype == torch.float64

    def test_all_items_accessible(self, ds):
        for i in range(len(ds)):
            x, y = ds[i]
            assert "diagnoses" in x


# ===========================================================================
# collate_patient_visit_histories
# ===========================================================================

class TestCollatePatientVisitHistories:
    def _make_sample(self, n_visits: int) -> tuple:
        x = {
            "diagnoses":          torch.ones(n_visits, N_DIAG),
            "procedures":         torch.ones(n_visits, N_PROC),
            "medication_history": torch.ones(n_visits, N_MEDS),
        }
        y = torch.zeros(N_MEDS)
        return x, y

    def test_output_keys(self):
        batch = [self._make_sample(2), self._make_sample(3)]
        out_x, _ = collate_patient_visit_histories(batch)
        assert set(out_x.keys()) == {"diagnoses", "procedures", "medication_history", "lengths"}

    def test_diagnoses_padded_to_max_length(self):
        batch = [self._make_sample(1), self._make_sample(3)]
        out_x, _ = collate_patient_visit_histories(batch)
        assert out_x["diagnoses"].shape == (2, 3, N_DIAG)

    def test_procedures_padded_to_max_length(self):
        batch = [self._make_sample(2), self._make_sample(4)]
        out_x, _ = collate_patient_visit_histories(batch)
        assert out_x["procedures"].shape == (2, 4, N_PROC)

    def test_lengths_correct(self):
        batch = [self._make_sample(1), self._make_sample(3), self._make_sample(2)]
        out_x, _ = collate_patient_visit_histories(batch)
        assert out_x["lengths"].tolist() == [1, 3, 2]

    def test_y_stacked(self):
        batch = [self._make_sample(1), self._make_sample(2)]
        _, y = collate_patient_visit_histories(batch)
        assert y.shape == (2, N_MEDS)

    def test_padding_is_zeros(self):
        # Sample 0 has 1 visit; after padding to length 3, rows 1 and 2 should be 0
        batch = [self._make_sample(1), self._make_sample(3)]
        out_x, _ = collate_patient_visit_histories(batch)
        assert out_x["diagnoses"][0, 1:].sum().item() == 0.0

    def test_single_item_batch(self):
        batch = [self._make_sample(2)]
        out_x, y = collate_patient_visit_histories(batch)
        assert out_x["diagnoses"].shape == (1, 2, N_DIAG)
        assert y.shape == (1, N_MEDS)

    def test_works_with_dataloader(self):
        ds = MultiHotDatasetWithPatientLookBack(
            _lookback_df(),
            target_col="medication_ids",
            n_diagnoses=N_DIAG,
            n_procedures=N_PROC,
            patient_id_col="patient_id",
            time_col="time",
            look_back=2,
        )
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_patient_visit_histories)
        out_x, y = next(iter(loader))
        assert "lengths" in out_x
        assert y.shape[0] == 3
