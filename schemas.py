from pydantic import BaseModel
from typing import Union

class PredictionInput(BaseModel):
    product_id: int
    defect_type: Union[str, int]
    defect_date: str
    defect_location: str
    repair_cost: float
    inspection_method: str

class PredictionResponse(BaseModel):
    Downtime: str
    Confidence: float
