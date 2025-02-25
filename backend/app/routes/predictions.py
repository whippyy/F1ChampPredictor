from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import PredictionRequest
from app.data_loader import load_csv_data
from datetime import datetime

router = APIRouter()
data = load_csv_data()

# Get current season data
current_year = datetime.now().year
current_races = data["races"][data["races"]["year"] == current_year]
valid_drivers = set(data["drivers"]["driverId"])
valid_tracks = set(current_races["circuitId"])

@router.post("/predict")
def predict_race_api(data: PredictionRequest):
    # 🚨 Check if driver is in the current season
    if data.driver_id not in valid_drivers:
        raise HTTPException(status_code=400, detail="Invalid driver for current season")

    # 🚨 Check if circuit is in the current season
    if data.circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    return predict_race(
        driver_id=data.driver_id,
        circuit_id=data.circuit_id,
        grid=data.grid,
        points=data.points,
        fastest_lap=data.fastest_lap,
        qualifying_position=data.qualifying_position,
        avg_qualifying_time=data.avg_qualifying_time
    )







