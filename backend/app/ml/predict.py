import tensorflow as tf
import numpy as np
import os
import joblib
from app.data_loader import load_csv_data
from datetime import datetime

MODEL_PATH = "app/ml/f1_model.keras"
SCALER_PATH = "app/ml/scaler.pkl"

# Load model & scaler
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# Load data for drivers & circuits
data = load_csv_data()
drivers_df = data["drivers"]
circuits_df = data["circuits"]

current_year = datetime.now().year

def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float, qualifying_position: int, avg_qualifying_time: float):
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # 🔹 Filter drivers and tracks for the current season
    current_races = data["races"][data["races"]["year"] == current_year]
    current_drivers = data["drivers"][data["drivers"]["driverId"].isin(current_races["raceId"])]

    if driver_id not in current_drivers["driverId"].values:
        return {"error": "This driver is not part of the current season."}

    if circuit_id not in current_races["circuitId"].values:
        return {"error": "This circuit is not part of the current season."}

    # 🔹 Get driver & track name
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    # 🔹 Prepare input data
    input_data = scaler.transform([[grid, points, fastest_lap, qualifying_position, avg_qualifying_time]])

    # 🔹 Predict position (convert back to 1-20 scale)
    predicted_position = int(model.predict(input_data)[0][0] * 20)
    predicted_position = max(1, min(predicted_position, 20))

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_position": predicted_position
    }
