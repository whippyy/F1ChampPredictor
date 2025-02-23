from pydantic import BaseModel

class PredictionInput(BaseModel):
    driver_id: int
    track_id: int
    grid: int
    points: float
    dob: int





