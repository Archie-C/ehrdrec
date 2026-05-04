import logging
import polars as pl

from ehrdrec.models.data_loading import LoadedData
from ehrdrec.models.data_processing import ProcessedDataMultiHot
from ehrdrec.processing.base import BaseProcessor
from ehrdrec.mappings import NDCATCMapper, Vocab

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MultiHotProcessor(BaseProcessor):

    def __init__(self):
        super().__init__()
        self.diagnoses_vocab   = None
        self.procedures_vocab  = None
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
    ) -> ProcessedDataMultiHot:
        logger.info(
            "Processing started [source=%r, dataset=%r, minimum_admissions=%d, "
            "split_frac=(%.2f, %.2f, %.2f), include_reserved=%s]",
            data.data_source, data.dataset_name,
            minimum_admissions, *split_frac, include_reserved,
        )

        df = data.frame
        logger.debug("LazyFrame received from LoadedData")

        logger.debug(
            "Step 1/6: applying patient filter [minimum_admissions=%d]",
            minimum_admissions,
        )
        df = self._filter_by_patient(df, minimum_admissions=minimum_admissions)

        logger.debug("Step 2/6: converting NDC codes to ATC codes [mapping_file=%r]", mapping_file)
        df = self._convert_ndc_to_atc(df, mapping_file=mapping_file)

        logger.debug(
            "Step 3/6: splitting data [train=%.2f, val=%.2f, test=%.2f]",
            *split_frac,
        )
        train_data, val_data, test_data = self._split(
            df,
            train_frac=split_frac[0],
            val_frac=split_frac[1],
            test_frac=split_frac[2],
        )

        logger.debug("Step 4/6: building vocabularies from training split")
        self._create_mappings(train_data)

        logger.debug("Step 5/6: encoding codes to integer IDs for all splits")
        train_data = self._convert_codes_to_integers(train_data)
        val_data   = self._convert_codes_to_integers(val_data)
        test_data  = self._convert_codes_to_integers(test_data)

        logger.debug(
            "Step 6/6: converting to multi-hot encoding [include_reserved=%s]",
            include_reserved,
        )
        train_data = self._convert_to_multihot(train_data, include_reserved=include_reserved)
        val_data   = self._convert_to_multihot(val_data,   include_reserved=include_reserved)
        test_data  = self._convert_to_multihot(test_data,  include_reserved=include_reserved)

        logger.info(
            "Processing complete [source=%r, dataset=%r]",
            data.data_source, data.dataset_name,
        )
        return ProcessedDataMultiHot(
            data_source=data.data_source,
            dataset_name=data.dataset_name,
            processor_type="multi_hot",
            train_frame=train_data,
            val_frame=val_data,
            test_frame=test_data,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _filter_by_patient(
        self, data: pl.LazyFrame, minimum_admissions: int = 1
    ) -> pl.LazyFrame:
        logger.debug("Building patient filter predicate [minimum_admissions=%d]", minimum_admissions)
        lf_filtered = (
            data
            .with_columns(
                pl.col("admission_id")
                .n_unique()
                .over("patient_id")
                .alias("n_admissions")
            )
            .filter(pl.col("n_admissions") >= minimum_admissions)
            .drop("n_admissions")
        )
        logger.debug("Patient filter predicate constructed; n_admissions helper column queued for drop")
        return lf_filtered

    def _convert_ndc_to_atc(
        self,
        data: pl.LazyFrame,
        mapping_file: str,
    ) -> pl.LazyFrame:
        logger.debug("Loading NDCATCMapper [file=%r]", mapping_file)
        mapper = NDCATCMapper.from_file(mapping_file)
        logger.debug("NDCATCMapper loaded")

        def map_meds_to_atcs(meds) -> list[str]:
            if meds is None:
                logger.debug("map_meds_to_atcs: null medication list; returning ['UNK']")
                return ["UNK"]

            meds = meds.to_list() if hasattr(meds, "to_list") else list(meds)
            logger.debug("map_meds_to_atcs: processing %d medication entries", len(meds))

            atcs: list[str] = []
            null_count = 0
            unmapped_count = 0

            for med in meds:
                if med is None:
                    null_count += 1
                    continue

                ndc = med.get("NDC") if isinstance(med, dict) else med["NDC"]

                if ndc is None or str(ndc).strip() in {"", "0"}:
                    logger.debug("map_meds_to_atcs: medication entry missing NDC field; skipping")
                    null_count += 1
                    continue

                mapped = mapper.ndc_to_atc(str(ndc))
                if not mapped:
                    logger.debug("map_meds_to_atcs: no ATC mapping found for NDC=%r; using 'UNK'", ndc)
                    unmapped_count += 1
                    mapped = ["UNK"]

                logger.debug("map_meds_to_atcs: NDC=%r -> %r", ndc, mapped)
                atcs.extend(mapped.atc_codes if mapped and mapped.atc_codes else ["UNK"])

            if null_count:
                logger.debug(
                    "map_meds_to_atcs: %d medication entr%s skipped due to null NDC",
                    null_count, "y" if null_count == 1 else "ies",
                )
            if unmapped_count:
                logger.warning(
                    "map_meds_to_atcs: %d NDC code%s had no ATC mapping and were substituted with 'UNK'",
                    unmapped_count, "" if unmapped_count == 1 else "s",
                )

            result = list(dict.fromkeys(atcs)) or ["UNK"]
            logger.debug(
                "map_meds_to_atcs: resolved %d unique ATC code(s) from %d input(s)",
                len(result), len(meds),
            )
            return result

        logger.debug("Attaching NDC->ATC map_elements expression to LazyFrame")
        return data.with_columns(
            pl.col("medications")
            .map_elements(map_meds_to_atcs, return_dtype=pl.List(pl.Utf8))
            .alias("atc_codes")
        )

    def _create_mappings(self, train_data: pl.LazyFrame) -> None:
        logger.debug("Building diagnoses vocabulary from training split")
        self.diagnoses_vocab = Vocab.from_lazyframe(train_data, col="diagnoses")
        logger.debug("Diagnoses vocabulary built [size=%d]", len(self.diagnoses_vocab.id_to_token))

        logger.debug("Building procedures vocabulary from training split")
        self.procedures_vocab = Vocab.from_lazyframe(train_data, col="procedures")
        logger.debug("Procedures vocabulary built [size=%d]", len(self.procedures_vocab.id_to_token))

        logger.debug("Building medications vocabulary from training split")
        self.medications_vocab = Vocab.from_lazyframe(train_data, col="atc_codes")
        logger.debug("Medications vocabulary built [size=%d]", len(self.medications_vocab.id_to_token))

        logger.info(
            "Vocabularies built [diagnoses=%d, procedures=%d, medications=%d]",
            len(self.diagnoses_vocab.id_to_token),
            len(self.procedures_vocab.id_to_token),
            len(self.medications_vocab.id_to_token),
        )

    def _convert_codes_to_integers(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Attaching integer-encoding expression: diagnoses -> diagnosis_ids")
        data = data.with_columns(self.diagnoses_vocab.encode_expr("diagnoses", "diagnosis_ids"))

        logger.debug("Attaching integer-encoding expression: procedures -> procedure_ids")
        data = data.with_columns(self.procedures_vocab.encode_expr("procedures", "procedure_ids"))

        logger.debug("Attaching integer-encoding expression: atc_codes -> atc_ids")
        data = data.with_columns(self.medications_vocab.encode_expr("atc_codes", "atc_ids"))

        return data

    def _convert_to_multihot(
        self, data: pl.LazyFrame, include_reserved: bool = True
    ) -> pl.LazyFrame:
        logger.debug("Attaching multi-hot expression: diagnosis_ids -> diagnosis_multihot")
        data = data.with_columns(
            self.diagnoses_vocab.to_multihot_expr(
                "diagnosis_ids", "diagnosis_multihot", include_reserved=include_reserved
            )
        )

        logger.debug("Attaching multi-hot expression: procedure_ids -> procedure_multihot")
        data = data.with_columns(
            self.procedures_vocab.to_multihot_expr(
                "procedure_ids", "procedure_multihot", include_reserved=include_reserved
            )
        )

        logger.debug("Attaching multi-hot expression: atc_ids -> medication_multihot")
        data = data.with_columns(
            self.medications_vocab.to_multihot_expr(
                "atc_ids", "medication_multihot", include_reserved=include_reserved
            )
        )

        cols_to_drop = [
            "diagnoses", "procedures", "atc_codes",
            "diagnosis_ids", "procedure_ids", "atc_ids",
            "medications",
        ]
        logger.debug("Dropping intermediate columns %s", cols_to_drop)
        return data.drop(cols_to_drop)

    def _split(
        self,
        data: pl.LazyFrame,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        logger.debug(
            "Split requested [train=%.2f, val=%.2f, test=%.2f]",
            train_frac, val_frac, test_frac,
        )
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, \
            "Split fractions must sum to 1.0."

        logger.debug("Sorting by admission_time and assigning row index")
        indexed = data.sort("admission_time").with_row_index("split_idx")

        logger.debug("Collecting row count to compute split boundaries")
        n = indexed.select(pl.len()).collect().item()
        logger.info("Total rows available for splitting: %d", n)

        train_end = int(n * train_frac)
        val_end   = train_end + int(n * val_frac)
        logger.debug(
            "Split boundaries [train=[0, %d), val=[%d, %d), test=[%d, %d)]",
            train_end, train_end, val_end, val_end, n,
        )

        if train_end == 0:
            logger.warning(
                "Train split is empty [n=%d, train_frac=%.2f]", n, train_frac
            )
        if val_end - train_end == 0:
            logger.warning(
                "Validation split is empty [n=%d, val_frac=%.2f]", n, val_frac
            )
        if n - val_end == 0:
            logger.warning(
                "Test split is empty [n=%d, test_frac=%.2f]", n, test_frac
            )

        logger.debug("Constructing split LazyFrames")
        train_lf = indexed.filter(pl.col("split_idx") < train_end).drop("split_idx")
        val_lf   = indexed.filter(
            (pl.col("split_idx") >= train_end) & (pl.col("split_idx") < val_end)
        ).drop("split_idx")
        test_lf  = indexed.filter(pl.col("split_idx") >= val_end).drop("split_idx")

        logger.debug("Split LazyFrames constructed (not yet collected)")
        return train_lf, val_lf, test_lf