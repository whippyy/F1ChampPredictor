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
def predict_entire_race(data: TrackPredictionRequest):
    """
    Predicts the entire race order for a given track.
    """
    circuit_id = data.circuit_id

    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    print(f"🚦 Predicting race for track {circuit_id}...")

    drivers_in_season = list(valid_drivers)
    raw_predictions = []

    for driver_id in drivers_in_season:
        grid_position, avg_lap_time = get_driver_stats(driver_id, circuit_id)

        # Get prediction
        prediction = predict_race(
            driver_id=driver_id,
            circuit_id=circuit_id,
            grid=grid_position,
        )

        # Skip failed predictions
        if prediction.get("status") != "success":
            continue

        raw_predictions.append((
            driver_id, 
            prediction["predicted_race_position"], 
            prediction
        ))

    # Sort by predicted position
    raw_predictions.sort(key=lambda x: x[1])

    # Assign final positions (in case of ties or missing predictions)
    predictions = []
    for position, (driver_id, _, pred) in enumerate(raw_predictions, start=1):
        driver_row = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
        team_id = results_df[
            (results_df["driverId"] == driver_id) & 
            (results_df["raceId"].isin(current_races["raceId"]))
        ]["constructorId"].values[0]
        team_row = constructors_df[constructors_df["constructorId"] == team_id].iloc[0]

        predictions.append({
            "position": position,
            "driver_id": driver_id,
            "driver": f"{driver_row['forename']} {driver_row['surname']}",
            "driver_code": driver_row.get("code", ""),
            "team": team_row["name"],
            "team_code": team_row.get("constructorRef", ""),
            "track": pred["track"]
        })

    return {
        "track": predictions[0]["track"] if predictions else "Unknown",
        "predictions": predictions
    }