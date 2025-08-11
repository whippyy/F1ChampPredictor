from pydantic import BaseModel, Field
from typing import List, Optional, Any

# --- Request Models ---
class TrackPredictionRequest(BaseModel):
    circuit_id: int

# --- Data Models ---
class Driver(BaseModel):
    driverId: int
    driverRef: str
    number: Optional[str] = None
    code: Optional[str] = None
    forename: str
    surname: str
    dob: str
    nationality: str
    url: str

class Team(BaseModel):
    constructorId: int
    constructorRef: str
    name: str
    nationality: str
    url: str

class Circuit(BaseModel):
    circuitId: int
    circuitRef: str
    name: str
    location: str
    country: str
    lat: float
    lng: float
    alt: Optional[Any] = None
    url: str

class Race(BaseModel):
    raceId: int
    year: int
    round: int
    circuitId: int
    name: str
    date: str
    time: Optional[str] = None
    url: Optional[str] = None

class Prediction(BaseModel):
    driver_id: int
    position: int
    driver_name: str
    team: str
    grid_position: int

class PredictionResponse(BaseModel):
    track: str
    predictions: List[Prediction]
