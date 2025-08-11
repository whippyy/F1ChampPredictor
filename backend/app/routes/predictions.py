from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import TrackPredictionRequest
from app.data_loader import f1_data
import numpy as np
import pandas as pd

router = APIRouter()

# --- Pre-computation and Data Optimization ---
# Set DataFrame indexes for faster lookups
races_df = f1_data.data["races"]
results_df = f1_data.data["results"]
drivers_df = f1_data.data["drivers"].set_index('driverId')
constructors_df = f1_data.data["constructors"].set_index('constructorId')
circuits_df = f1_data.data["circuits"].set_index('circuitId')

# Determine the current year dynamically from the latest race
current_year = races_df['year'].max()
current_races = races_df[races_df["year"] == current_year]
valid_tracks = set(current_races["circuitId"])

# Get valid drivers and their most recent team affiliation for the current season
current_results = results_df[results_df["raceId"].isin(current_races["raceId"])]
valid_drivers = set(current_results["driverId"])

# Create a mapping of driverId to their latest constructorId for the current year
driver_team_map = current_results.sort_values('raceId').drop_duplicates('driverId', keep='last').set_index('driverId')['constructorId']

# --- Helper Functions ---

def get_last_grid_position(driver_id: int, results_df: pd.DataFrame) -> int:
    """Get the last known grid position for a driver in the current season."""
    driver_results = results_df[results_df["driverId"] == driver_id]
    if not driver_results.empty:
        # Sort by raceId to get the most recent race
        return driver_results.sort_values(by="raceId", ascending=False)["grid"].iloc[0]
    # Default to a mid-pack position if no data is available
    return 10

@router.post("/predict-race")
def predict_entire_race(data: TrackPredictionRequest):
    circuit_id = data.circuit_id
    
    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail=f"Invalid circuit for {current_year} season")
    
    predictions = []
    
    for driver_id in valid_drivers:
        try:
            # 1. Get Last Grid Position
            grid = get_last_grid_position(driver_id, current_results)
            grid = int(max(1, min(20, grid))) # Ensure grid is within 1-20
            
            # 2. Call the ML Model
            prediction_result = predict_race(
                driver_id=driver_id,
                circuit_id=circuit_id,
                grid=grid
            )
            
            if prediction_result["status"] == "success":
                # 3. Get Driver and Team Info (using efficient lookups)
                driver_info = drivers_df.loc[driver_id]
                team_id = driver_team_map.get(driver_id)
                team_info = constructors_df.loc[team_id] if team_id else {"name": "Unknown"}

                predictions.append({
                    "driver_id": int(driver_id),
                    # The model's raw output is used for sorting
                    "position": int(prediction_result["predicted_race_position"]),
                    "driver_name": f"{driver_info['forename']} {driver_info['surname']}",
                    "team": team_info["name"],
                    "grid_position": grid
                })
                
        except Exception as e:
            # Log the error but continue processing other drivers
            print(f"⚠️ Error processing driver {driver_id}: {str(e)}")
            continue
    
    # 4. Sort predictions based on the model's output
    # This treats the model as a "ranker". The raw predicted position determines the
    # finishing order, but we re-number them sequentially from 1 to N to ensure
    # a clean, gap-free, and duplicate-free final result for the frontend.
    predictions.sort(key=lambda x: x["position"])
    
    # 5. Final position adjustment for a clean 1-N list
    for i, p in enumerate(predictions, 1):
        p["position"] = i
    
    return {
        "track": circuits_df.loc[circuit_id]["name"],
        "predictions": predictions
    }