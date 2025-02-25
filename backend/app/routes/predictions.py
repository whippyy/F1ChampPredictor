from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import PredictionRequest
from app.data_loader import load_csv_data
from datetime import datetime

router = APIRouter()
data = load_csv_data()

# ✅ Get current season data
current_year = 2024
current_races = data["races"][data["races"]["year"] == current_year]

# ✅ Debugging: Ensure these contain valid values
print(f"🟢 Current Season: {current_year}")
print(f"🟢 Valid Circuits for {current_year}: {current_races['circuitId'].unique().tolist()}")

valid_drivers = set(data["results"][data["results"]["raceId"].isin(current_races["raceId"])]["driverId"])
valid_tracks = set(current_races["circuitId"])

print(f"🟢 Valid Drivers for {current_year}: {list(valid_drivers)}")
print(f"🟢 Valid Tracks for {current_year}: {list(valid_tracks)}")

@router.post("/predict")
def predict_race_api(data: PredictionRequest):
    print(f"📥 Received request: {data}")

    # 🚨 Debugging: Show received values before checking
    print(f"🔍 Checking driver {data.driver_id} in valid drivers: {valid_drivers}")
    print(f"🔍 Checking circuit {data.circuit_id} in valid circuits: {valid_tracks}")

    # 🚨 Check if driver is in the current season
    if data.driver_id not in valid_drivers:
        print("❌ Invalid driver!")
        raise HTTPException(status_code=400, detail="Invalid driver for current season")

    # 🚨 Check if circuit is in the current season
    if data.circuit_id not in valid_tracks:
        print("❌ Invalid circuit!")
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    print("✅ Driver & Circuit are valid. Proceeding with prediction...")
    
    return predict_race(
        driver_id=data.driver_id,
        circuit_id=data.circuit_id,
        grid=data.grid,
        points=data.points,
        fastest_lap=data.fastest_lap,
        qualifying_position=data.qualifying_position,
        avg_qualifying_time=data.avg_qualifying_time
    )









