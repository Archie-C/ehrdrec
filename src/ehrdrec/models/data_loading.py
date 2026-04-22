from pydantic import BaseModel
from typing import List

from ehrdrec.models.base import Medication

class LoadedDataRow(BaseModel):
    patient_id: str
    admission_id: str
    admission_time: str
    discharge_time: str
    diagnoses: List[str]
    procedures: List[str]
    medications: List[Medication]
    

class LoadedData(BaseModel):
    data_source: str
    dataset_name: str
    data: List[LoadedDataRow]