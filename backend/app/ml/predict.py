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

# ✅ Compute average lap time per driver per race
avg_lap_time = lap_times_df.groupby(["raceId", "driverId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)

# ✅ Merge it into results_df
results_df = results_df.merge(avg_lap_time, on=["raceId", "driverId"], how="left")

print("✅ Merged avg_lap_time into results_df:", results_df.columns)


# ✅ Ensure all numeric columns are properly converted
numeric_columns = ["milliseconds", "grid", "points", "fastestLapSpeed", "avg_lap_time"]  # Add other relevant columns
for col in numeric_columns:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")  # Convert invalid values to NaN
    results_df[col].fillna(results_df[col].median(), inplace=True)  # Replace NaNs with median values


# ✅ Get current season (2024)
current_season = 2024
current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
current_races = races_df[races_df["year"] == current_season]

# ✅ Get valid drivers & circuits for 2024
valid_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]["driverId"].unique()
valid_circuits = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float):
    """
    Predicts the final race result with track-specific data.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Ensure input features match the trained model
    feature_names = ["grid", "points", "fastestLapSpeed", "avg_lap_time"]


    lap_times_df["milliseconds"] = pd.to_numeric(lap_times_df["milliseconds"], errors="coerce")
    lap_times_df["milliseconds"].fillna(lap_times_df["milliseconds"].median(), inplace=True)

    # ✅ Ensure `milliseconds` is numeric (fixes the TypeError issue)
    results_df["milliseconds"] = pd.to_numeric(results_df["milliseconds"], errors="coerce")
    
    # ✅ Compute track-level average lap time
    track_avg_lap_time = results_df[results_df["raceId"].isin(
        races_df[races_df["circuitId"] == circuit_id]["raceId"]
    )]["milliseconds"].mean()

    # ✅ Compute driver-specific average lap time
    driver_avg_lap_time = results_df[
        (results_df["driverId"] == driver_id) & (results_df["raceId"].isin(current_races["raceId"]))
    ]["milliseconds"].mean()

    # ✅ Use driver lap time if available, otherwise fallback to track average
    avg_lap_time = driver_avg_lap_time if not np.isnan(driver_avg_lap_time) else track_avg_lap_time

    # ✅ If still NaN, use the median lap time instead of a hardcoded default
    if np.isnan(avg_lap_time):
        avg_lap_time = results_df["milliseconds"].median()

    print(f"📊 Avg Lap Time for Driver {driver_id} on Track {circuit_id}: {avg_lap_time}")

    # ✅ Create input array
    input_data = pd.DataFrame([[grid, points, fastest_lap, avg_lap_time]], columns=feature_names)

    # ✅ Transform input data
    input_data_scaled = scaler.transform(input_data)

    # ✅ Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    predicted_position = max(1, min(round(predicted_position), 20))  # Ensure within range

    # ✅ Get driver & track names
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_race_position": predicted_position
    }
