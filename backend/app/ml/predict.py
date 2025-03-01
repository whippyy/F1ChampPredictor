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

# Ensure circuitId exists in qualifying_df before merging
if "circuitId" not in qualifying_df.columns:
    qualifying_df = qualifying_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Convert qualifying times (q1, q2, q3) to numeric values (in milliseconds)
for col in ["q1", "q2", "q3"]:
    qualifying_df[col] = pd.to_numeric(qualifying_df[col], errors="coerce")

# Calculate average qualifying time
qualifying_df["avg_qualifying_time"] = qualifying_df[["q1", "q2", "q3"]].mean(axis=1)

# Ensure numeric columns are properly converted
numeric_columns = ["grid", "fastestLapSpeed", "avg_lap_time"]
for col in numeric_columns:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df[col].fillna(results_df[col].median(), inplace=True)

def predict_race(driver_id: int, circuit_id: int, grid: int):
    """
    Predicts the final race result with track-specific data.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # Fetch driver info
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"
    driver_code = driver_row["code"].values[0] if "code" in driver_row.columns and not driver_row.empty else None

    # Fetch team info
    team_id = results_df[results_df["driverId"] == driver_id]["constructorId"].values[0] if "constructorId" in results_df.columns else None
    team_row = data["constructors"][data["constructors"]["constructorId"] == team_id] if team_id else None
    team_name = team_row["name"].values[0] if team_row is not None and not team_row.empty else "Unknown Team"
    team_code = team_row["constructorRef"].values[0] if team_row is not None and "constructorRef" in team_row.columns else None

    # Compute driver-specific stats
    driver_avg_lap_time = results_df[
        (results_df["driverId"] == driver_id) & (results_df["circuitId"] == circuit_id)
    ]["avg_lap_time"].mean()

    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df[results_df["circuitId"] == circuit_id]["avg_lap_time"].mean()
    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df["avg_lap_time"].median()

    # Prepare input for model
    input_data = pd.DataFrame([[grid, driver_avg_lap_time]],
                              columns=["grid", "avg_lap_time"])

    input_data_scaled = scaler.transform(input_data)
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    predicted_position = max(1, min(round(predicted_position), 20)) if not np.isnan(predicted_position) else 10

    return {
        "driver": driver_name,
        "driver_code": driver_code,
        "team": team_name,
        "team_code": team_code,
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0] if circuit_id in circuits_df["circuitId"].values else "Unknown Track",
        "predicted_race_position": predicted_position
    }
