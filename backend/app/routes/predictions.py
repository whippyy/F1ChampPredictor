from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import TrackPredictionRequest
from app.data_loader import load_csv_data
import numpy as np
import pandas as pd

router = APIRouter()
data = load_csv_data()

# ✅ Load current season data
current_year = 2024
races_df = data["races"]
results_df = data["results"]
drivers_df = data["drivers"]
constructors_df = data["constructors"]

# ✅ Filter current season races
current_races = races_df[races_df["year"] == current_year]
valid_tracks = set(current_races["circuitId"])
valid_drivers = set(results_df[results_df["raceId"].isin(current_races["raceId"])]["driverId"])

print(f"🟢 Current Season: {current_year}")
print(f"🟢 Valid Circuits for {current_year}: {valid_tracks}")
print(f"🟢 Valid Drivers for {current_year}: {valid_drivers}")

def get_driver_stats(driver_id, circuit_id):
    """
    Fetch past grid position and average lap time for a driver.
    """
    driver_results = results_df[
        (results_df["driverId"] == driver_id) & (results_df["raceId"].isin(current_races["raceId"]))
    ]
    
    # ✅ Get last race grid position (default to 10 if no data)
    last_race_grid_position = (
        driver_results.sort_values(by="raceId", ascending=False)["grid"].values[0] 
        if not driver_results.empty else 10
    )

    # ✅ Compute average lap time (use median if no valid data)
    avg_lap_time = driver_results["milliseconds"].mean()
    avg_lap_time = avg_lap_time if not np.isnan(avg_lap_time) else results_df["milliseconds"].median()

    print(f"📊 Driver {driver_id}: Grid={last_race_grid_position}, Avg Lap={avg_lap_time}")

    return last_race_grid_position, avg_lap_time


@router.post("/predict-race")
@router.post("/predict-race")
@router.post("/predict-race")
def predict_entire_race(data: TrackPredictionRequest):
    circuit_id = data.circuit_id
    
    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    # Get all drivers who participated in current season
    current_drivers = results_df[
        results_df["raceId"].isin(current_races["raceId"])
    ]["driverId"].unique()
    
    predictions = []
    for driver_id in current_drivers:
        # Get most recent grid position for this driver
        last_race = results_df[
            (results_df["driverId"] == driver_id) &
            (results_df["raceId"].isin(current_races["raceId"]))
        ].sort_values("raceId").iloc[-1]
        
        grid = last_race["grid"]
        
        prediction = predict_race(
            driver_id=driver_id,
            circuit_id=circuit_id,
            grid=grid
        )
        
        if prediction["status"] == "success":
            driver_info = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
            team_info = constructors_df[
                constructors_df["constructorId"] == last_race["constructorId"]
            ].iloc[0]
            
            predictions.append({
                "driver_id": driver_id,
                "position": prediction["predicted_race_position"],
                "driver_name": f"{driver_info['forename']} {driver_info['surname']}",
                "team": team_info["name"],
                "grid_position": grid
            })
    
    # Sort by predicted position
    predictions.sort(key=lambda x: x["position"])
    
    return {
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0],
        "predictions": predictions
    }