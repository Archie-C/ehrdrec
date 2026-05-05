import hashlib
import json
import logging
from pathlib import Path

import polars as pl
from platformdirs import user_cache_dir

from ehrdrec.models.data_loading import LoadedData
from ehrdrec.models.data_processing import ProcessedDataMultiHot
from ehrdrec.processing.base import BaseProcessor
from ehrdrec.mappings import NDCATCMapper, Vocab

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MultiHotProcessor(BaseProcessor):

    PROCESSOR_VERSION = 1

    def __init__(self, cache_dir: Path | None = None):
        super().__init__()
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(user_cache_dir("ehrdrec")) / "processors"
        )

        self.diagnoses_vocab = None
        self.procedures_vocab = None
        self.medications_vocab = None

        logger.debug("Initialised; vocab attributes set to None")

    def process(
        self,
        data: LoadedData,
        *,
        minimum_admissions: int = 1,
        split_frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
        mapping_file: str = "data/mappings/ndc_atc_mapping.sqlite",
        include_reserved: bool = True,
        force_reload: bool = False,
    ) -> ProcessedDataMultiHot:

        cache_dir = self._cache_dir(
            data=data,
            minimum_admissions=minimum_admissions,
            split_frac=split_frac,
            mapping_file=mapping_file,
            include_reserved=include_reserved,
        )

        if not force_reload and self._cache_exists(cache_dir):
            try:
                logger.info("Loading MultiHotProcessor output from cache: %s", cache_dir)
                self._load_vocabs(cache_dir)

                return ProcessedDataMultiHot(
                    data_source=data.data_source,
                    dataset_name=data.dataset_name,
                    processor_type="multi_hot",
                    train_frame=pl.scan_parquet(cache_dir / "train.parquet"),
                    val_frame=pl.scan_parquet(cache_dir / "val.parquet"),
                    test_frame=pl.scan_parquet(cache_dir / "test.parquet"),
                )

            except Exception:
                logger.warning(
                    "Failed to load MultiHotProcessor cache; rebuilding",
                    exc_info=True,
                )

        logger.info(
            "Processing started [source=%r, dataset=%r, minimum_admissions=%d, "
            "split_frac=(%.2f, %.2f, %.2f), include_reserved=%s]",
            data.data_source,
            data.dataset_name,
            minimum_admissions,
            *split_frac,
            include_reserved,
        )

        df = data.frame

        df = self._filter_by_patient(
            df,
            minimum_admissions=minimum_admissions,
        )

        df = self._convert_ndc_to_atc(
            df,
            mapping_file=mapping_file,
        )

        train_data, val_data, test_data = self._split(
            df,
            train_frac=split_frac[0],
            val_frac=split_frac[1],
            test_frac=split_frac[2],
        )

        self._create_mappings(train_data)

        train_data = self._convert_codes_to_integers(train_data)
        val_data = self._convert_codes_to_integers(val_data)
        test_data = self._convert_codes_to_integers(test_data)

        train_data = self._convert_to_multihot(
            train_data,
            include_reserved=include_reserved,
        )
        val_data = self._convert_to_multihot(
            val_data,
            include_reserved=include_reserved,
        )
        test_data = self._convert_to_multihot(
            test_data,
            include_reserved=include_reserved,
        )

        self._write_cache(cache_dir, train_data, val_data, test_data)

        logger.info("Processing complete; cached output to: %s", cache_dir)

        return ProcessedDataMultiHot(
            data_source=data.data_source,
            dataset_name=data.dataset_name,
            processor_type="multi_hot",
            train_frame=pl.scan_parquet(cache_dir / "train.parquet"),
            val_frame=pl.scan_parquet(cache_dir / "val.parquet"),
            test_frame=pl.scan_parquet(cache_dir / "test.parquet"),
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_dir(
        self,
        *,
        data: LoadedData,
        minimum_admissions: int,
        split_frac: tuple[float, float, float],
        mapping_file: str,
        include_reserved: bool,
    ) -> Path:
        key = self._cache_key(
            data=data,
            minimum_admissions=minimum_admissions,
            split_frac=split_frac,
            mapping_file=mapping_file,
            include_reserved=include_reserved,
        )
        return self.cache_dir / f"multi_hot_{key}"

    def _cache_key(
        self,
        *,
        data: LoadedData,
        minimum_admissions: int,
        split_frac: tuple[float, float, float],
        mapping_file: str,
        include_reserved: bool,
    ) -> str:
        mapping_path = Path(mapping_file)

        payload = {
            "processor": "multi_hot",
            "processor_version": self.PROCESSOR_VERSION,
            "data_source": str(data.data_source),
            "dataset_name": data.dataset_name,
            "minimum_admissions": minimum_admissions,
            "split_frac": list(split_frac),
            "include_reserved": include_reserved,
            "mapping_file": str(mapping_path.resolve())
            if mapping_path.exists()
            else str(mapping_file),
            "mapping_mtime": mapping_path.stat().st_mtime
            if mapping_path.exists()
            else None,
            "mapping_size": mapping_path.stat().st_size
            if mapping_path.exists()
            else None,
        }

        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _cache_exists(cache_dir: Path) -> bool:
        required = [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "diagnoses_vocab.json",
            "procedures_vocab.json",
            "medications_vocab.json",
            "meta.json",
        ]
        return cache_dir.exists() and all((cache_dir / f).exists() for f in required)

    def _write_cache(
        self,
        cache_dir: Path,
        train_data: pl.LazyFrame,
        val_data: pl.LazyFrame,
        test_data: pl.LazyFrame,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)

        train_data.collect().write_parquet(
            cache_dir / "train.parquet",
            compression="zstd",
            compression_level=3,
        )
        val_data.collect().write_parquet(
            cache_dir / "val.parquet",
            compression="zstd",
            compression_level=3,
        )
        test_data.collect().write_parquet(
            cache_dir / "test.parquet",
            compression="zstd",
            compression_level=3,
        )

        self._save_vocabs(cache_dir)

        meta = {
            "processor": "multi_hot",
            "processor_version": self.PROCESSOR_VERSION,
        }

        (cache_dir / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    def _save_vocabs(self, cache_dir: Path) -> None:
        self._save_vocab(cache_dir / "diagnoses_vocab.json", self.diagnoses_vocab)
        self._save_vocab(cache_dir / "procedures_vocab.json", self.procedures_vocab)
        self._save_vocab(cache_dir / "medications_vocab.json", self.medications_vocab)

    def _load_vocabs(self, cache_dir: Path) -> None:
        self.diagnoses_vocab = self._load_vocab(cache_dir / "diagnoses_vocab.json")
        self.procedures_vocab = self._load_vocab(cache_dir / "procedures_vocab.json")
        self.medications_vocab = self._load_vocab(cache_dir / "medications_vocab.json")

    @staticmethod
    def _save_vocab(path: Path, vocab: Vocab) -> None:
        data = {
            "token_to_id": vocab.token_to_id,
            "id_to_token": vocab.id_to_token,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _load_vocab(path: Path) -> Vocab:
        data = json.loads(path.read_text(encoding="utf-8"))

        vocab = Vocab.__new__(Vocab)
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = data["id_to_token"]

        return vocab

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def _filter_by_patient(
        self,
        data: pl.LazyFrame,
        minimum_admissions: int = 1,
    ) -> pl.LazyFrame:
        return (
            data.with_columns(
                pl.col("admission_id")
                .n_unique()
                .over("patient_id")
                .alias("n_admissions")
            )
            .filter(pl.col("n_admissions") >= minimum_admissions)
            .drop("n_admissions")
        )

    def _convert_ndc_to_atc(
        self,
        data: pl.LazyFrame,
        mapping_file: str,
    ) -> pl.LazyFrame:
        mapper = NDCATCMapper.from_file(mapping_file)

        ndc_cache: dict[str, list[str]] = {}

        def cached_ndc_to_atc(ndc: str) -> list[str]:
            if ndc in ndc_cache:
                return ndc_cache[ndc]

            mapped = mapper.ndc_to_atc(ndc)

            if not mapped or not mapped.atc_codes:
                result = ["UNK"]
            else:
                result = list(mapped.atc_codes)

            ndc_cache[ndc] = result
            return result

        def map_meds_to_atcs(meds) -> list[str]:
            if meds is None:
                return ["UNK"]

            meds = meds.to_list() if hasattr(meds, "to_list") else list(meds)

            atcs: list[str] = []

            for med in meds:
                if med is None:
                    continue

                ndc = med.get("NDC") if isinstance(med, dict) else med["NDC"]

                if ndc is None or str(ndc).strip() in {"", "0"}:
                    continue

                atcs.extend(cached_ndc_to_atc(str(ndc).strip()))

            return list(dict.fromkeys(atcs)) or ["UNK"]

        return data.with_columns(
            pl.col("medications")
            .map_elements(map_meds_to_atcs, return_dtype=pl.List(pl.Utf8))
            .alias("atc_codes")
        )

    def _create_mappings(self, train_data: pl.LazyFrame) -> None:
        self.diagnoses_vocab = Vocab.from_lazyframe(train_data, col="diagnoses")
        self.procedures_vocab = Vocab.from_lazyframe(train_data, col="procedures")
        self.medications_vocab = Vocab.from_lazyframe(train_data, col="atc_codes")

        logger.info(
            "Vocabularies built [diagnoses=%d, procedures=%d, medications=%d]",
            len(self.diagnoses_vocab.id_to_token),
            len(self.procedures_vocab.id_to_token),
            len(self.medications_vocab.id_to_token),
        )

    def _convert_codes_to_integers(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            [
                self.diagnoses_vocab.encode_expr("diagnoses", "diagnosis_ids"),
                self.procedures_vocab.encode_expr("procedures", "procedure_ids"),
                self.medications_vocab.encode_expr("atc_codes", "atc_ids"),
            ]
        )

    def _convert_to_multihot(
        self,
        data: pl.LazyFrame,
        include_reserved: bool = True,
    ) -> pl.LazyFrame:
        data = data.with_columns(
            [
                self.diagnoses_vocab.to_multihot_expr(
                    "diagnosis_ids",
                    "diagnosis_multihot",
                    include_reserved=include_reserved,
                ),
                self.procedures_vocab.to_multihot_expr(
                    "procedure_ids",
                    "procedure_multihot",
                    include_reserved=include_reserved,
                ),
                self.medications_vocab.to_multihot_expr(
                    "atc_ids",
                    "medication_multihot",
                    include_reserved=include_reserved,
                ),
            ]
        )

        return data.drop(
            [
                "diagnoses",
                "procedures",
                "atc_codes",
                "diagnosis_ids",
                "procedure_ids",
                "atc_ids",
                "medications",
            ]
        )

    def _split(
        self,
        data: pl.LazyFrame,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, (
            "Split fractions must sum to 1.0."
        )

        indexed = data.sort("admission_time").with_row_index("split_idx")

        n = indexed.select(pl.len()).collect().item()

        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        train_lf = indexed.filter(pl.col("split_idx") < train_end).drop("split_idx")

        val_lf = indexed.filter(
            (pl.col("split_idx") >= train_end)
            & (pl.col("split_idx") < val_end)
        ).drop("split_idx")

        test_lf = indexed.filter(pl.col("split_idx") >= val_end).drop("split_idx")

        return train_lf, val_lf, test_lf