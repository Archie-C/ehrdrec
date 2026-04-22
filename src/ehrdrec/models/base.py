from pydantic import BaseModel

class Medication(BaseModel):
    id: str
    
class ExtendedMedication(Medication):
    name: str
    dosage_value: str
    doseage_unit: str