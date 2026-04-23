from dataclasses import dataclass
import polars as pl

@dataclass(slots=True)
class ProcessedDataMultiHot:
    """
    Raw MIMIC-III data as a single flat LazyFrame, ready for downstream
    Polars processing.
 
    Schema
    ------
    patient_id      : int
    admission_id    : int
    admission_time  : datetime
    discharge_time  : datetime
    diagnoses       : List[int]
    procedures      : List[int]
    medications     : List[int]
    """
    data_source: str
    dataset_name: str
    processor_type: str
    frame: pl.LazyFrame