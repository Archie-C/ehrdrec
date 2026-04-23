from dataclasses import dataclass
import polars as pl

@dataclass(slots=True)
class LoadedData:
    """
    Raw MIMIC-III data as a single flat LazyFrame, ready for downstream
    Polars processing.
 
    Schema
    ------
    patient_id      : Utf8
    admission_id    : Utf8
    admission_time  : Utf8   (ISO-8601, "" when null)
    discharge_time  : Utf8   (ISO-8601, "" when null)
    diagnoses       : List[Utf8]
    procedures      : List[Utf8]
    medications     : List[Struct{ NDC, name, dosage_value, dosage_unit }]
    """
    data_source: str
    dataset_name: str
    frame: pl.LazyFrame