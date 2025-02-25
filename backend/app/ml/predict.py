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
qualifying_df = data["qualifying"]

# ✅ Convert qualifying lap times to milliseconds
def time_to_milliseconds(time_str):
    """Convert '1:26.572' to total milliseconds, safely handling missing values."""
    if pd.isna(time_str) or time_str in ["\\N", "", None, "nan"]:
        return np.nan  # Return NaN for missing values

    time_str = str(time_str).strip()  # Convert to string and remove extra spaces
    if ":" not in time_str:
        print(f"⚠️ Invalid time format encountered: {time_str}")
        return np.nan  # Invalid format

    try:
        minutes, seconds = time_str.split(":")
        return int(minutes) * 60000 + float(seconds) * 1000
    except ValueError:
        print(f"⚠️ Could not convert time: {time_str}")
        return np.nan

# ✅ Preprocess qualifying data
qualifying_df.replace("\\N", np.nan, inplace=True)
for col in ["q1", "q2", "q3"]:
    qualifying_df[col] = qualifying_df[col].apply(time_to_milliseconds)
qualifying_df["avg_qualifying_time"] = qualifying_df[["q1", "q2", "q3"]].mean(axis=1)

print("✅ Processed Qualifying Data:")
print(qualifying_df[["driverId", "raceId", "avg_qualifying_time"]].head(10))

# ✅ Get current season (2024)
current_season = 2024
current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
current_races = races_df[races_df["year"] == current_season]

# ✅ Get valid drivers & circuits for 2024
valid_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]["driverId"].unique()
valid_circuits = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

def predict_qualifying_position(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float):
    """
    Predicts the qualifying position for a driver on a given track using real past data.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    print("✅ Scaler was fitted with features:", scaler.feature_names_in_)

    # ✅ Fetch qualifying times for driver
    qualifying_times = qualifying_df[
        (qualifying_df["driverId"] == driver_id) & 
        (qualifying_df["raceId"].isin(current_races["raceId"]))
    ][["avg_qualifying_time"]]

    avg_qualifying_time = qualifying_times["avg_qualifying_time"].mean() if not qualifying_times.empty else 90000
    print(f"📊 Final Avg Qualifying Time for Driver {driver_id}: {avg_qualifying_time}")

    # ✅ Ensure feature names match trained model
    feature_names = ["grid", "points", "fastestLapSpeed", "avg_qualifying_time", "qualifying_position"]
    input_data = pd.DataFrame([[grid, points, fastest_lap, avg_qualifying_time, 0]],
                              columns=feature_names)
    input_data_scaled = scaler.transform(input_data)

    # ✅ Predict qualifying position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    print(f"🔍 Predicted Qualifying Position for Driver {driver_id}: {predicted_position}")

    return max(1, min(round(predicted_position), 20))  # Ensure within 1-20 range

def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float, qualifying_position: int, avg_qualifying_time: float):
    """
    Predicts the final race result after determining the qualifying position.
    Returns a dictionary with driver details.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Ensure feature names match trained model
    feature_names = ["grid", "points", "fastestLapSpeed", "avg_qualifying_time", "qualifying_position"]
    input_data = pd.DataFrame([[grid, points, fastest_lap, avg_qualifying_time, qualifying_position]],
                              columns=feature_names)
    input_data_scaled = scaler.transform(input_data)

    # ✅ Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20

    predicted_position = max(1, min(round(predicted_position), 20))  # Ensure between 1-20

    # ✅ Get driver & track name
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_qualifying_position": qualifying_position,
        "predicted_race_position": predicted_position
    }
