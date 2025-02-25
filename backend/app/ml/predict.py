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

# ✅ Get current season (2024)
current_season = 2024
current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]

# ✅ Get valid drivers & circuits for 2024
valid_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]["driverId"].unique()
valid_circuits = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

def predict_qualifying_position(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float, avg_qualifying_time: float):
    """
    Predicts the qualifying position for a driver on a given track.
    Ensures input data is valid and prevents NaN errors.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Ensure 7 features are passed into the model
    input_data = scaler.transform([[grid, points, fastest_lap, avg_qualifying_time, circuit_id, driver_id, 0]])

    # 🚨 Debugging: Check for NaN values after scaling
    if np.isnan(input_data).any():
        print(f"❌ NaN detected in input data: {input_data}")
        return 10  # Default to mid-grid position if NaN

    # ✅ Predict position
    predicted_position = model.predict(input_data)[0][0] * 20

    # 🚨 Handle NaN results
    if np.isnan(predicted_position):
        print("❌ NaN detected in model output. Defaulting to position 10.")
        return 10  # Default to mid-grid if NaN

    return max(1, min(round(predicted_position), 20))  # Ensure 1-20 range



def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float, qualifying_position: int, avg_qualifying_time: float):
    """
    Predicts the final race result after determining the qualifying position.
    Returns a dictionary with driver details.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Ensure 7 features are passed to match the trained model
    input_data = scaler.transform([[grid, points, fastest_lap, qualifying_position, avg_qualifying_time, circuit_id, driver_id]])

    # 🚨 Debugging: Check for NaN values after scaling
    if np.isnan(input_data).any():
        print(f"❌ NaN detected in input data: {input_data}")
        predicted_position = 10  # Default to mid-grid position if NaN
    else:
        predicted_position = model.predict(input_data)[0][0] * 20

        # 🚨 Handle NaN results
        if np.isnan(predicted_position):
            print("❌ NaN detected in model output. Defaulting to position 10.")
            predicted_position = 10  # Default to mid-grid if NaN

    predicted_position = max(1, min(round(predicted_position), 20))  # Ensure between 1-20

    # ✅ Get driver & track name
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    # ✅ Always return a dictionary
    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_qualifying_position": qualifying_position,
        "predicted_race_position": predicted_position
    }

