from typing import Optional

from pydantic import BaseModel

class Medication(BaseModel):
    id: str
    
class ExtendedMedication(Medication):
    name: Optional[str] = None
    dosage_value: Optional[str] = None
    dosage_unit: Optional[str] = None