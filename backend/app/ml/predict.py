import pandas as pd
import tensorflow as tf
import numpy as np
import os
import joblib
from app.data_loader import load_csv_data

MODEL_PATH = "app/ml/f1_model.keras"
SCALER_PATH = "app/ml/scaler.pkl"

# Load model & scaler
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# Load data
data = load_csv_data()
drivers_df = data["drivers"]
circuits_df = data["circuits"]
races_df = data["races"]
results_df = data["results"]
lap_times_df = data["lap_times"]
pit_stops_df = data["pit_stops"]
qualifying_df = data["qualifying"]

# Ensure circuitId exists in lap_times_df before merging
if "circuitId" not in lap_times_df.columns:
    lap_times_df = lap_times_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Compute driver’s average lap time per circuit
avg_lap_time = lap_times_df.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)

# Ensure circuitId exists in results_df before merging
if "circuitId" not in results_df.columns:
    results_df = results_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Merge avg_lap_time into results_df
results_df = results_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

# Ensure numeric columns are properly converted
numeric_columns = ["grid", "fastestLapSpeed", "avg_lap_time"]
for col in numeric_columns:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df[col].fillna(results_df[col].median(), inplace=True)

def predict_race(driver_id: int, circuit_id: int, grid: int, fastest_lap: float):
    """
    Predicts the final race result with track-specific data.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # Fetch driver-specific average lap time at the specific track
    driver_avg_lap_time = results_df[
        (results_df["driverId"] == driver_id) & (results_df["circuitId"] == circuit_id)
    ]["avg_lap_time"].mean()

    # Fallback to track-wide average if driver-specific data is missing
    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df[results_df["circuitId"] == circuit_id]["avg_lap_time"].mean()

    # Ensure valid lap time
    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df["avg_lap_time"].median()

    # Fetch driver-specific average qualifying time at the specific track
    driver_avg_qualifying_time = qualifying_df[
        (qualifying_df["driverId"] == driver_id) & (qualifying_df["circuitId"] == circuit_id)
    ]["avg_qualifying_time"].mean()

    # Fallback to track-wide average if driver-specific data is missing
    if np.isnan(driver_avg_qualifying_time):
        driver_avg_qualifying_time = qualifying_df[qualifying_df["circuitId"] == circuit_id]["avg_qualifying_time"].mean()

    # Ensure valid qualifying time
    if np.isnan(driver_avg_qualifying_time):
        driver_avg_qualifying_time = qualifying_df["avg_qualifying_time"].median()

    print(f"📊 Driver {driver_id} - Avg Lap Time at Circuit {circuit_id}: {driver_avg_lap_time}")
    print(f"📊 Driver {driver_id} - Avg Qualifying Time at Circuit {circuit_id}: {driver_avg_qualifying_time}")

    # Prepare input data with track-specific features
    input_data = pd.DataFrame([[grid, fastest_lap, driver_avg_lap_time, driver_avg_qualifying_time]], 
                              columns=["grid", "fastestLapSpeed", "avg_lap_time", "avg_qualifying_time"])
    
    # Transform input data
    input_data_scaled = scaler.transform(input_data)

    # Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    predicted_position = max(1, min(round(predicted_position), 20))  # Ensure within range

    # Get driver & track names
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_race_position": predicted_position
    }