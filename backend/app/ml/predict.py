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

    # Check if a team exists for this driver, and handle cases where the team might not exist
    if team_id is not None:
        team_row = data["constructors"][data["constructors"]["constructorId"] == team_id]
    else:
        team_row = None

    team_name = team_row["name"].values[0] if team_row is not None and not team_row.empty else "Unknown Team"
    team_code = team_row["constructorRef"].values[0] if team_row is not None and "constructorRef" in team_row.columns else None

    # Fetch driver-specific average lap time at the specific track
    driver_avg_lap_time = results_df[
        (results_df["driverId"] == driver_id) & (results_df["circuitId"] == circuit_id)
    ]["avg_lap_time"].mean()

    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df[results_df["circuitId"] == circuit_id]["avg_lap_time"].mean()
    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = results_df["avg_lap_time"].median()

    # Fetch driver-specific average qualifying time
    driver_avg_qualifying_time = qualifying_df[
        (qualifying_df["driverId"] == driver_id) & (qualifying_df["circuitId"] == circuit_id)
    ]["avg_qualifying_time"].mean()

    if np.isnan(driver_avg_qualifying_time):
        driver_avg_qualifying_time = qualifying_df[qualifying_df["circuitId"] == circuit_id]["avg_qualifying_time"].mean()
    if np.isnan(driver_avg_qualifying_time):
        driver_avg_qualifying_time = qualifying_df["avg_qualifying_time"].median()

    # Fetch driver-specific average pit stop time
    driver_avg_pit_time = pit_stops_df[
        (pit_stops_df["driverId"] == driver_id) & (pit_stops_df["raceId"].isin(races_df[races_df["circuitId"] == circuit_id]["raceId"]))
    ]["milliseconds"].mean()

    if np.isnan(driver_avg_pit_time):
        driver_avg_pit_time = pit_stops_df["milliseconds"].median()

    # Fetch driver-specific grid position
    grid_position = results_df[
        (results_df["driverId"] == driver_id) & (results_df["circuitId"] == circuit_id)
    ]["grid"].mean()

    if np.isnan(grid_position):
        grid_position = results_df["grid"].median()

    # ✅ Create input data with matching features
    input_data = pd.DataFrame([[grid_position, driver_avg_lap_time, driver_avg_pit_time, driver_avg_qualifying_time]],
                              columns=["grid_position", "avg_lap_time", "avg_pit_time", "avg_qualifying_time"])

    print(f"🛠️ Model Input Data: {input_data}")
    print(f"📢 Expected Features During Training: {scaler.feature_names_in_}")

    # Transform input data
    input_data_scaled = scaler.transform(input_data)

    # Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    if np.isnan(predicted_position):
        print("⚠️ Model returned NaN. Assigning default position (10).")
        predicted_position = 10

    predicted_position = max(1, min(round(predicted_position), 20))

    # Ensure the predicted result is within range and corresponding team information is updated
    result = {
        "driver_id": driver_id,  # Include driver_id in the result
        "driver": driver_name,
        "driver_code": driver_code,
        "team": team_name,
        "team_code": team_code,
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0] if circuit_id in circuits_df["circuitId"].values else "Unknown Track",
        "predicted_race_position": predicted_position
    }

    return result
