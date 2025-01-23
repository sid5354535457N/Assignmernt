from pydantic import BaseModel

class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

class PredictionResponse(BaseModel):
    Downtime: str
    Confidence: float
