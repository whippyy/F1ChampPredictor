from pydantic import BaseModel

class PredictionRequest(BaseModel):
    driver_id: int
    circuit_id: int
    grid: int
    points: float
    fastest_lap: float
    qualifying_position: int
    avg_qualifying_time: float


class TrackPredictionRequest(BaseModel):
    circuit_id: int






