from pydantic import BaseModel

class PredictionInput(BaseModel):
    driver_id: int
    circuit_id: int
    grid: int
    points: float
    dob: int
    fastest_lap: float





