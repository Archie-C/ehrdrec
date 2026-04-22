from pydantic import BaseModel

class Medication(BaseModel):
    id: str
    
class ExtendedMedication(Medication):
    name: str
    dosage: str