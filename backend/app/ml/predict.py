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

# Debugging output to check valid circuits
print("🟢 Valid circuits for 2024:", valid_circuits)

def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float, qualifying_position: int, avg_qualifying_time: float):
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Check if driver is valid for 2024
    if driver_id not in valid_drivers:
        print(f"❌ Driver {driver_id} is not in valid drivers for 2024: {valid_drivers}")
        return {"error": "Invalid driver for current season"}

    # ✅ Check if circuit is valid for 2024
    print(f"🔍 Checking circuit_id={circuit_id} (type: {type(circuit_id)}) against valid circuits...")

    # Convert circuit_id to int before checking
    if int(circuit_id) not in set(map(int, valid_circuits)):
        print(f"❌ Circuit {circuit_id} is NOT in valid circuits: {valid_circuits}")
        return {"error": "Invalid circuit for current season"}



    # ✅ Get driver & track name
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    # ✅ Prepare input
    input_data = scaler.transform([[grid, points, fastest_lap, qualifying_position, avg_qualifying_time]])

    # ✅ Predict position (convert back to 1-20 scale)
    predicted_position = int(model.predict(input_data)[0][0] * 20)
    predicted_position = max(1, min(predicted_position, 20))

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_position": predicted_position
    }
