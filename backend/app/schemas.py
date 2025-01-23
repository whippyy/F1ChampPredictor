# app/schemas.py
from pydantic import BaseModel
from typing import List

class DriverBase(BaseModel):
    name: str
    team: str
    points: float

class DriverCreate(DriverBase):
    pass

class Driver(DriverBase):
    id: int

    class Config:
        orm_mode = True  # This tells Pydantic to read data as dictionaries

class PredictionRequest(BaseModel):
    driver_id: int  # e.g., ID of the driver
    team_id: int    # e.g., ID of the team
    race_id: int    # e.g., ID of the race
    historical_data: List[float]  # any historical stats or features needed for prediction


