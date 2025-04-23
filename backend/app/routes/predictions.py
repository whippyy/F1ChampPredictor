from fastapi import APIRouter, HTTPException
from app.ml.predict import predict_race
from app.schemas import TrackPredictionRequest
from app.data_loader import f1_data
import numpy as np

router = APIRouter()

# Get all data from centralized loader
races_df = f1_data.data["races"]
results_df = f1_data.data["results"]
drivers_df = f1_data.data["drivers"]
constructors_df = f1_data.data["constructors"]
circuits_df = f1_data.data["circuits"]

current_year = 2024
current_races = races_df[races_df["year"] == current_year]
valid_tracks = set(current_races["circuitId"])
valid_drivers = set(results_df[results_df["raceId"].isin(current_races["raceId"])]["driverId"])

def get_driver_stats(driver_id):
    driver_results = results_df[
        (results_df["driverId"] == driver_id) & 
        (results_df["raceId"].isin(current_races["raceId"]))
    ]
    return (
        driver_results.sort_values(by="raceId", ascending=False)["grid"].values[0]
        if not driver_results.empty else 10
    )

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

@router.post("/predict-race")
def predict_entire_race(data: TrackPredictionRequest):
    circuit_id = data.circuit_id
    
    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")
    
    predictions = []
    seen_positions = set()  # To track used positions
    
    for driver_id in valid_drivers:
        try:
            grid = get_driver_stats(driver_id)
            # Ensure grid position is valid (1-20)
            grid = max(1, min(20, grid))
            
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
                
                # Get predicted position and ensure it's unique
                predicted_pos = int(prediction["predicted_race_position"])
                while predicted_pos in seen_positions:
                    predicted_pos += 1
                seen_positions.add(predicted_pos)
                
                # Ensure position is between 1-20
                predicted_pos = max(1, min(20, predicted_pos))
                
                predictions.append({
                    "driver_id": int(driver_id),
                    "position": predicted_pos,
                    "driver_name": f"{driver_info['forename']} {driver_info['surname']}",
                    "team": team_info["name"],
                    "grid_position": int(grid)
                })
                
        except Exception as e:
            print(f"⚠️ Error processing driver {driver_id}: {str(e)}")
            continue
    
    # Sort by position and ensure no duplicates
    predictions.sort(key=lambda x: x["position"])
    
    # Final position adjustment to ensure no gaps or duplicates
    for i, prediction in enumerate(predictions, 1):
        prediction["position"] = i
    
    return {
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0],
        "predictions": predictions
    }