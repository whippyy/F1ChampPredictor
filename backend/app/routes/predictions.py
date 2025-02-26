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

# ✅ Filter current season races
current_races = races_df[races_df["year"] == current_year]
valid_tracks = set(current_races["circuitId"])
valid_drivers = set(results_df[results_df["raceId"].isin(current_races["raceId"])] ["driverId"])

print(f"🟢 Current Season: {current_year}")
print(f"🟢 Valid Circuits for {current_year}: {valid_tracks}")
print(f"🟢 Valid Drivers for {current_year}: {valid_drivers}")

def get_driver_stats(driver_id, circuit_id):
    """
    Fetches past grid position and previous points for a driver.
    """
    # ✅ Get previous points
    driver_results = results_df[
        (results_df["driverId"] == driver_id) & (results_df["raceId"].isin(current_races["raceId"]))
    ]
    previous_points = driver_results["points"].sum()

    # ✅ Get last race grid position
    last_race_grid_position = (
        driver_results.sort_values(by="raceId", ascending=False)["grid"].values[0] if not driver_results.empty else 10
    )

    return last_race_grid_position, previous_points

@router.post("/predict-race")
def predict_entire_race(data: TrackPredictionRequest):
    """
    Predicts the entire race order for a given track.
    """
    circuit_id = data.circuit_id

    # 🚨 Ensure circuit is valid for the current season
    if circuit_id not in valid_tracks:
        raise HTTPException(status_code=400, detail="Invalid circuit for current season")

    # ✅ Fetch valid drivers for the season
    drivers_in_season = list(valid_drivers)
    raw_predictions = []

    # ✅ Loop through all drivers and predict race position
    for driver_id in drivers_in_season:
        # ✅ Fetch past grid position and points
        grid_position, previous_points = get_driver_stats(driver_id, circuit_id)

        # ✅ Predict race position
        prediction = predict_race(
            driver_id=driver_id,
            circuit_id=circuit_id,
            grid=grid_position,
            points=previous_points,
            fastest_lap=90.0  # Placeholder value
        )

        raw_predictions.append((driver_id, prediction["predicted_race_position"], prediction))

    # ✅ Sort by race position
    raw_predictions.sort(key=lambda x: x[1])

    # ✅ Assign unique positions
    for i, (_, _, prediction) in enumerate(raw_predictions):
        prediction["predicted_race_position"] = i + 1

    # ✅ Return results
    predictions = [prediction for _, _, prediction in raw_predictions]

    return {
        "track": predictions[0]["track"],
        "predictions": predictions
    }

