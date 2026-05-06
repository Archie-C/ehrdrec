from dataclasses import dataclass
import polars as pl

@dataclass(slots=True)
class ProcessedData:
    data_source: str
    dataset_name: str
    processor_type: str
    train_frame: pl.LazyFrame
    val_frame: pl.LazyFrame
    test_frame: pl.LazyFrame

@dataclass(slots=True)
class ProcessedDataMultiHot(ProcessedData):
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
    pass