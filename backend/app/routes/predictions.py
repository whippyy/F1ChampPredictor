from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import TrackPredictionRequest
from app.data_loader import f1_data
import numpy as np

router = APIRouter()

# Get data from centralized loader
races_df = f1_data.data["races"]
results_df = f1_data.data["results"]
drivers_df = f1_data.data["drivers"]
constructors_df = f1_data.data["constructors"]

# Filter current season races
current_year = 2024
current_races = races_df[races_df["year"] == current_year]
valid_tracks = set(current_races["circuitId"])
valid_drivers = set(results_df[results_df["raceId"].isin(current_races["raceId"])]["driverId"])

def get_driver_stats(driver_id):
    """Fetch past grid position and average lap time for a driver."""
    driver_results = results_df[
        (results_df["driverId"] == driver_id) & 
        (results_df["raceId"].isin(current_races["raceId"]))
    ]
    
    # Get last race grid position (default to 10 if no data)
    last_race_grid_position = (
        driver_results.sort_values(by="raceId", ascending=False)["grid"].values[0] 
        if not driver_results.empty else 10
    )

    return last_race_grid_position

@router.post("/predict-race")
def predict_entire_race(data: TrackPredictionRequest):
    circuit_id = data.circuit_id
    
    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    predictions = []
    for driver_id in valid_drivers:
        grid = get_driver_stats(driver_id)
        
        prediction = predict_race(
            driver_id=driver_id,
            circuit_id=circuit_id,
            grid=grid
        )
        
        if prediction["status"] == "success":
            driver_info = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
            team_info = constructors_df[
                constructors_df["constructorId"] == results_df[
                    (results_df["driverId"] == driver_id) &
                    (results_df["raceId"].isin(current_races["raceId"]))
                ].iloc[-1]["constructorId"]
            ].iloc[0]
            
            predictions.append({
                "driver_id": driver_id,
                "position": prediction["predicted_race_position"],
                "driver_name": f"{driver_info['forename']} {driver_info['surname']}",
                "team": team_info["name"],
                "grid_position": grid
            })
    
    # Sort by predicted position and assign final positions
    predictions.sort(key=lambda x: x["position"])
    for i, prediction in enumerate(predictions, start=1):
        prediction["position"] = i
    
    return {
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0],
        "predictions": predictions
    }