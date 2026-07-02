"""
Tests for ehrdrec.processing and ehrdrec.evaluation.

Processing strategy: the top-level process() method requires a live SQLite
mapping file and writes to disk, so we test the private helper methods
directly via synthetic LazyFrames. Each helper is a pure LazyFrame transform.

Evaluation strategy: build a trivial nn.Module and a DataLoader from a
synthetic dataset, then run the Evaluator against known metrics.
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import polars as pl
from torch.utils.data import DataLoader, TensorDataset

from ehrdrec.processing.to_multihot.to_multihot import MultiHotProcessor
from ehrdrec.processing.to_multihot.to_multihot_many_atc import MultiHotProcessorAllATCs
from ehrdrec.processing.to_multihot.llm_codes import LLMCodeProcessor
from ehrdrec.evaluation.standard import Evaluator
from ehrdrec.metrics.jaccard import Jaccard
from ehrdrec.metrics.f1 import F1
from ehrdrec.metrics.prauc import PRAUC
from ehrdrec.mappings.code_to_id.vocab import Vocab


# ===========================================================================
# Shared synthetic LazyFrame
# ===========================================================================

def _base_lf(n: int = 10) -> pl.LazyFrame:
    """Minimal LazyFrame that matches the expected processor input schema."""
    return pl.LazyFrame({
        "patient_id":    [str(i % 3) for i in range(n)],
        "admission_id":  [str(i)     for i in range(n)],
        "admission_time": [f"2020-01-{i+1:02d}" for i in range(n)],
        "diagnoses":     [["D001", "D002"] if i % 2 == 0 else ["D003"] for i in range(n)],
        "procedures":    [["P001"] if i % 3 == 0 else ["P002", "P003"] for i in range(n)],
        "atc_codes":     [["A10B", "C01A"] if i % 2 == 0 else ["N06A"] for i in range(n)],
        "medications":   [[]    for _ in range(n)],  # not used in helper-level tests
    })


def _processor_with_vocabs() -> MultiHotProcessor:
    """Return a MultiHotProcessor whose vocabs are pre-built from _base_lf."""
    p = MultiHotProcessor.__new__(MultiHotProcessor)
    p.cache_dir = Path("/tmp/ehrdrec_test_cache")

    lf = _base_lf()
    p.diagnoses_vocab   = Vocab.from_lazyframe(lf, "diagnoses")
    p.procedures_vocab  = Vocab.from_lazyframe(lf, "procedures")
    p.medications_vocab = Vocab.from_lazyframe(lf, "atc_codes")

    return p


# ===========================================================================
# MultiHotProcessor — private helpers
# ===========================================================================

class TestMultiHotProcessorHelpers:

    @pytest.fixture
    def proc(self):
        return _processor_with_vocabs()

    @pytest.fixture
    def lf(self):
        return _base_lf()

    # --- _filter_by_patient ---

    def test_filter_minimum_1_keeps_all(self, proc, lf):
        out = proc._filter_by_patient(lf, minimum_admissions=1).collect()
        assert out.height == 10

    def test_filter_minimum_2_removes_singles(self, proc, lf):
        # patient "0" appears at rows 0,3,6,9 (4 rows); "1" at 1,4,7 (3 rows);
        # "2" at 2,5,8 (3 rows). All have ≥ 2 admissions.
        out = proc._filter_by_patient(lf, minimum_admissions=2).collect()
        assert out.height == 10  # all patients have ≥ 2

    def test_filter_high_threshold_removes_patients(self, proc):
        # Build a frame where patient "X" has only 1 admission
        lf = pl.LazyFrame({
            "patient_id":    ["X", "Y", "Y"],
            "admission_id":  ["a", "b", "c"],
            "admission_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "diagnoses":     [["D001"], ["D001"], ["D002"]],
            "procedures":    [["P001"], ["P001"], ["P002"]],
            "atc_codes":     [["A10B"], ["A10B"], ["C01A"]],
            "medications":   [[], [], []],
        })
        out = proc._filter_by_patient(lf, minimum_admissions=2).collect()
        # Patient X should be removed
        assert "X" not in out["patient_id"].to_list()
        assert out.height == 2

    # --- _split ---

    def test_split_sizes_sum_to_total(self, proc, lf):
        train, val, test = proc._split(lf, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        total = train.collect().height + val.collect().height + test.collect().height
        assert total == 10

    def test_split_fractions_approximate(self, proc):
        lf = _base_lf(100)
        train, val, test = proc._split(lf, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        assert train.collect().height == 80
        assert val.collect().height == 10
        assert test.collect().height == 10

    def test_split_ordered_by_admission_time(self, proc, lf):
        train, val, test = proc._split(lf, train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_times = train.collect()["admission_time"].to_list()
        all_times   = lf.sort("admission_time").collect()["admission_time"].to_list()
        # Train set should hold the earliest rows
        assert train_times == sorted(train_times)
        assert all(t <= all_times[8] for t in train_times)

    def test_split_fracs_not_summing_to_1_raises(self, proc, lf):
        with pytest.raises(AssertionError):
            proc._split(lf, train_frac=0.5, val_frac=0.3, test_frac=0.1)

    # --- _convert_codes_to_integers ---

    def test_encode_diagnoses_column_present(self, proc, lf):
        out = proc._convert_codes_to_integers(lf).collect()
        assert "diagnosis_ids" in out.columns

    def test_encode_procedures_column_present(self, proc, lf):
        out = proc._convert_codes_to_integers(lf).collect()
        assert "procedure_ids" in out.columns

    def test_encode_medications_column_present(self, proc, lf):
        out = proc._convert_codes_to_integers(lf).collect()
        assert "atc_ids" in out.columns

    def test_encoded_ids_are_integers(self, proc, lf):
        out = proc._convert_codes_to_integers(lf).collect()
        first_diag_ids = out["diagnosis_ids"][0].to_list()
        assert all(isinstance(i, int) for i in first_diag_ids)

    def test_unknown_code_encodes_as_unk(self, proc):
        lf = pl.LazyFrame({
            "patient_id":    ["1"],
            "admission_id":  ["a"],
            "admission_time": ["2020-01-01"],
            "diagnoses":     [["ZZZZ"]],
            "procedures":    [["P001"]],
            "atc_codes":     [["A10B"]],
            "medications":   [[]],
        })
        out = proc._convert_codes_to_integers(lf).collect()
        assert out["diagnosis_ids"][0].to_list() == [0]  # UNK = 0

    # --- _convert_to_multihot ---

    def test_multihot_column_present(self, proc, lf):
        encoded = proc._convert_codes_to_integers(lf)
        out = proc._convert_to_multihot(encoded).collect()
        assert "medication_multihot" in out.columns

    def test_multihot_vector_length(self, proc, lf):
        encoded = proc._convert_codes_to_integers(lf)
        out = proc._convert_to_multihot(encoded).collect()
        vec = out["medication_multihot"][0].to_list()
        assert len(vec) == proc.medications_vocab.vocab_size

    def test_multihot_values_binary(self, proc, lf):
        encoded = proc._convert_codes_to_integers(lf)
        out = proc._convert_to_multihot(encoded).collect()
        for row in out["medication_multihot"].to_list():
            assert all(v in (0, 1) for v in row)

    def test_multihot_drops_intermediate_columns(self, proc, lf):
        encoded = proc._convert_codes_to_integers(lf)
        out = proc._convert_to_multihot(encoded).collect()
        assert "atc_codes" not in out.columns
        assert "atc_ids"   not in out.columns
        assert "diagnoses" not in out.columns

    # --- _cache_exists ---

    def test_cache_not_exists_for_empty_dir(self, tmp_path):
        p = MultiHotProcessor(cache_dir=tmp_path)
        assert not MultiHotProcessor._cache_exists(tmp_path / "nonexistent")

    def test_cache_exists_when_all_files_present(self, tmp_path):
        required = [
            "train.parquet", "val.parquet", "test.parquet",
            "diagnoses_vocab.json", "procedures_vocab.json",
            "medications_vocab.json", "meta.json",
        ]
        d = tmp_path / "cache"
        d.mkdir()
        for f in required:
            (d / f).write_text("{}")
        assert MultiHotProcessor._cache_exists(d)

    def test_cache_not_exists_when_file_missing(self, tmp_path):
        required = [
            "train.parquet", "val.parquet", "test.parquet",
            "diagnoses_vocab.json", "procedures_vocab.json",
            # missing: medications_vocab.json and meta.json
        ]
        d = tmp_path / "cache"
        d.mkdir()
        for f in required:
            (d / f).write_text("{}")
        assert not MultiHotProcessor._cache_exists(d)

    # --- vocab save / load round-trip ---

    def test_vocab_save_load_roundtrip(self, proc, tmp_path):
        path = tmp_path / "vocab.json"
        MultiHotProcessor._save_vocab(path, proc.diagnoses_vocab)
        loaded = MultiHotProcessor._load_vocab(path)
        assert loaded.token_to_id == proc.diagnoses_vocab.token_to_id
        assert loaded.id_to_token == proc.diagnoses_vocab.id_to_token

    def test_saved_vocab_is_valid_json(self, proc, tmp_path):
        path = tmp_path / "vocab.json"
        MultiHotProcessor._save_vocab(path, proc.diagnoses_vocab)
        data = json.loads(path.read_text())
        assert "token_to_id" in data
        assert "id_to_token" in data


# ===========================================================================
# LLMCodeProcessor — keeps symbolic codes plus metric columns
# ===========================================================================

class TestLLMCodeProcessorHelpers:

    @pytest.fixture
    def proc(self):
        p = LLMCodeProcessor.__new__(LLMCodeProcessor)
        p.cache_dir = Path("/tmp/ehrdrec_test_cache_llm_codes")

        lf = _base_lf()
        p.diagnoses_vocab = Vocab.from_lazyframe(lf, "diagnoses")
        p.procedures_vocab = Vocab.from_lazyframe(lf, "procedures")
        p.medications_vocab = Vocab.from_lazyframe(lf, "atc_codes")
        return p

    @pytest.fixture
    def encoded(self, proc):
        return proc._convert_codes_to_integers(_base_lf())

    def test_add_metric_columns_keeps_symbolic_codes(self, proc, encoded):
        out = proc._add_metric_columns(encoded).collect()
        for col in ("diagnoses", "procedures", "atc_codes"):
            assert col in out.columns

    def test_add_metric_columns_keeps_id_columns_for_metrics(self, proc, encoded):
        out = proc._add_metric_columns(encoded).collect()
        for col in ("diagnosis_ids", "procedure_ids", "atc_ids"):
            assert col in out.columns

    def test_add_metric_columns_adds_medication_multihot(self, proc, encoded):
        out = proc._add_metric_columns(encoded).collect()
        vec = out["medication_multihot"][0].to_list()
        assert len(vec) == proc.medications_vocab.vocab_size
        assert all(v in (0, 1) for v in vec)

    def test_add_metric_columns_drops_raw_medication_structs(self, proc, encoded):
        out = proc._add_metric_columns(encoded).collect()
        assert "medications" not in out.columns

    def test_cache_key_changes_with_atc_level(self, proc):
        from ehrdrec.models.dataclasses.data_loading import LoadedData

        data = LoadedData(
            data_source="synthetic",
            dataset_name="synthetic",
            frame=_base_lf(),
        )
        key_atc3 = proc._cache_key(
            data=data,
            minimum_admissions=1,
            split_frac=(0.8, 0.1, 0.1),
            mapping_file="missing.sqlite",
            include_reserved=True,
            atc_level=3,
        )
        key_atc5 = proc._cache_key(
            data=data,
            minimum_admissions=1,
            split_frac=(0.8, 0.1, 0.1),
            mapping_file="missing.sqlite",
            include_reserved=True,
            atc_level=5,
        )
        assert key_atc3 != key_atc5


# ===========================================================================
# MultiHotProcessorAllATCs — _add_atc_level_columns
# ===========================================================================

class TestMultiHotProcessorAllATCsHelpers:

    @pytest.fixture
    def proc(self):
        p = MultiHotProcessorAllATCs.__new__(MultiHotProcessorAllATCs)
        p.cache_dir = Path("/tmp/ehrdrec_test_cache_all_atc")

        lf = _base_lf()
        lf_with_atc_levels = p._add_atc_level_columns(lf)

        p.diagnoses_vocab   = Vocab.from_lazyframe(lf, "diagnoses")
        p.procedures_vocab  = Vocab.from_lazyframe(lf, "procedures")
        p.atc5_vocab        = Vocab.from_lazyframe(lf, "atc_codes")
        p.atc4_vocab        = Vocab.from_lazyframe(lf_with_atc_levels, "atc4_codes")
        p.atc3_vocab        = Vocab.from_lazyframe(lf_with_atc_levels, "atc3_codes")
        p.atc2_vocab        = Vocab.from_lazyframe(lf_with_atc_levels, "atc2_codes")
        p.atc1_vocab        = Vocab.from_lazyframe(lf_with_atc_levels, "atc1_codes")
        p.medications_vocab = p.atc5_vocab
        return p

    @pytest.fixture
    def lf_with_levels(self, proc):
        return proc._add_atc_level_columns(_base_lf())

    def test_atc_level_columns_added(self, proc, lf_with_levels):
        cols = lf_with_levels.collect().columns
        for level in ("atc1_codes", "atc2_codes", "atc3_codes", "atc4_codes", "atc5_codes"):
            assert level in cols

    def test_atc1_is_one_char(self, proc, lf_with_levels):
        out = lf_with_levels.collect()
        for codes in out["atc1_codes"].to_list():
            for code in codes:
                if code != "UNK":
                    assert len(code) == 1

    def test_atc2_is_three_chars(self, proc, lf_with_levels):
        out = lf_with_levels.collect()
        for codes in out["atc2_codes"].to_list():
            for code in codes:
                if code != "UNK":
                    assert len(code) == 3

    def test_atc3_is_four_chars(self, proc, lf_with_levels):
        out = lf_with_levels.collect()
        for codes in out["atc3_codes"].to_list():
            for code in codes:
                if code != "UNK":
                    assert len(code) == 4

    def test_atc5_matches_original(self, proc, lf_with_levels):
        out = lf_with_levels.collect()
        orig = _base_lf().collect()
        # atc5_codes should match the original atc_codes (both 7-char ATC5)
        for atc5_row, orig_row in zip(out["atc5_codes"].to_list(), orig["atc_codes"].to_list()):
            assert sorted(atc5_row) == sorted(orig_row)

    def test_atc1_is_prefix_of_atc5(self, proc, lf_with_levels):
        out = lf_with_levels.collect()
        for atc1_row, atc5_row in zip(out["atc1_codes"].to_list(), out["atc5_codes"].to_list()):
            for atc5_code in atc5_row:
                if atc5_code != "UNK":
                    expected_atc1 = atc5_code[:1]
                    assert expected_atc1 in atc1_row

    def test_null_atc_codes_produce_unk(self, proc):
        lf = pl.LazyFrame({"atc_codes": [None]}, schema={"atc_codes": pl.List(pl.Utf8)})
        out = proc._add_atc_level_columns(lf).collect()
        assert out["atc1_codes"][0].to_list() == ["UNK"]

    def test_encode_all_atc_level_columns(self, proc, lf_with_levels):
        out = proc._convert_codes_to_integers(lf_with_levels).collect()
        for col in ("atc1_ids", "atc2_ids", "atc3_ids", "atc4_ids", "atc5_ids"):
            assert col in out.columns

    def test_multihot_all_atc_levels_present(self, proc, lf_with_levels):
        encoded = proc._convert_codes_to_integers(lf_with_levels)
        out = proc._convert_to_multihot(encoded).collect()
        for col in ("atc1_multihot", "atc2_multihot", "atc3_multihot",
                    "atc4_multihot", "atc5_multihot", "medication_multihot"):
            assert col in out.columns

    def test_medication_multihot_is_alias_for_atc5(self, proc, lf_with_levels):
        encoded = proc._convert_codes_to_integers(lf_with_levels)
        out = proc._convert_to_multihot(encoded).collect()
        for atc5, med in zip(out["atc5_multihot"].to_list(), out["medication_multihot"].to_list()):
            assert atc5 == med


# ===========================================================================
# Evaluator
# ===========================================================================

class _ConstantModel(nn.Module):
    """Always returns the same logit tensor regardless of input."""
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        batch = x.shape[0] if isinstance(x, torch.Tensor) else next(iter(x.values())).shape[0]
        return {"predictions": self._logits.expand(batch, -1)}


def _make_loader(inputs: torch.Tensor, targets: torch.Tensor, batch_size: int = 4) -> DataLoader:
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


N_MEDS = 8


class TestEvaluator:

    def test_run_returns_evaluation_results(self):
        logits  = torch.zeros(N_MEDS)
        targets = torch.zeros(4, N_MEDS)
        model   = _ConstantModel(logits)
        loader  = _make_loader(torch.rand(4, 10), targets)
        metrics = [Jaccard(ignore_indices=[], from_logits=True)]
        ev = Evaluator(model, loader, metrics=metrics, device="cpu")
        results = ev.run()
        assert hasattr(results, "test_metrics")

    def test_metric_names_in_results(self):
        model  = _ConstantModel(torch.zeros(N_MEDS))
        loader = _make_loader(torch.rand(4, 10), torch.zeros(4, N_MEDS))
        metrics = [Jaccard(ignore_indices=[]), F1(ignore_indices=[])]
        ev = Evaluator(model, loader, metrics=metrics, device="cpu")
        results = ev.run()
        assert "Jaccard" in results.test_metrics
        assert "F1"      in results.test_metrics

    def test_perfect_model_jaccard_is_one(self):
        # Model outputs large positive logits for slots 0 and 2.
        # Targets match exactly.
        logits  = torch.tensor([5.0, -5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        targets = torch.tensor([[1.0, 0.0, 1.0, 0.0,  0.0,  0.0,  0.0,  0.0]])
        model   = _ConstantModel(logits)
        loader  = _make_loader(torch.rand(1, 10), targets, batch_size=1)
        metrics = [Jaccard(ignore_indices=[], from_logits=True)]
        ev = Evaluator(model, loader, metrics=metrics, device="cpu")
        results = ev.run()
        assert results.test_metrics["Jaccard"] == pytest.approx(1.0)

    def test_no_metrics_raises(self):
        model  = _ConstantModel(torch.zeros(N_MEDS))
        loader = _make_loader(torch.rand(4, 10), torch.zeros(4, N_MEDS))
        ev = Evaluator(model, loader, metrics=[], device="cpu")
        with pytest.raises(ValueError, match="No metrics"):
            ev.run()

    def test_metrics_reset_between_runs(self):
        # Run twice; the second run should produce the same result as the first,
        # not an accumulated value.
        logits  = torch.tensor([5.0] + [-5.0] * (N_MEDS - 1))
        targets = torch.tensor([[1.0] + [0.0] * (N_MEDS - 1)])
        model   = _ConstantModel(logits)
        loader  = _make_loader(torch.rand(1, 10), targets, batch_size=1)
        metrics = [Jaccard(ignore_indices=[], from_logits=True)]
        ev = Evaluator(model, loader, metrics=metrics, device="cpu")
        r1 = ev.run()
        r2 = ev.run()
        assert r1.test_metrics["Jaccard"] == pytest.approx(r2.test_metrics["Jaccard"])

    def test_model_put_in_eval_mode(self):
        model  = _ConstantModel(torch.zeros(N_MEDS))
        model.train()
        loader = _make_loader(torch.rand(4, 10), torch.zeros(4, N_MEDS))
        ev = Evaluator(model, loader, metrics=[F1(ignore_indices=[])], device="cpu")
        ev.run()
        assert not model.training

    def test_multiple_batches_accumulated(self):
        # 8 identical samples in two batches of 4 → same result as one batch of 8
        logits  = torch.tensor([5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
        targets = torch.ones(8, N_MEDS) * torch.tensor([1,0,0,0,0,0,0,0], dtype=torch.float32)
        model   = _ConstantModel(logits)
        loader  = _make_loader(torch.rand(8, 10), targets, batch_size=4)
        metrics = [F1(ignore_indices=[], from_logits=True)]
        ev = Evaluator(model, loader, metrics=metrics, device="cpu")
        results = ev.run()
        assert results.test_metrics["F1"] == pytest.approx(1.0)

    def test_dict_input_model(self):
        # Evaluator should handle dict inputs by moving each value to device
        class DictInputModel(nn.Module):
            def forward(self, x):
                b = x["a"].shape[0]
                return {"predictions": torch.zeros(b, N_MEDS)}

        inputs  = {"a": torch.rand(4, 5), "b": torch.rand(4, 3)}
        targets = torch.zeros(4, N_MEDS)

        # Build a custom loader that yields dict inputs
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, i):
                return {"a": torch.rand(5), "b": torch.rand(3)}, torch.zeros(N_MEDS)

        loader  = DataLoader(DictDataset(), batch_size=4)
        metrics = [Jaccard(ignore_indices=[], from_logits=False)]
        ev = Evaluator(DictInputModel(), loader, metrics=metrics, device="cpu")
        results = ev.run()
        assert "Jaccard" in results.test_metrics
