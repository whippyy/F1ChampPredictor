import tensorflow as tf
import numpy as np
import os
from app.data_loader import load_csv_data

MODEL_PATH = "app/ml/f1_model.keras"

# Load model if exists
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    model = None
    print("⚠️ Model file not found! Train the model first.")

# Load data for drivers & tracks
data = load_csv_data()
drivers_df = data["drivers"]
races_df = data["races"]

def predict_race(driver_id: int, track_id: int, grid: int, points: float, dob: int):
    if model is None:
        raise ValueError("No trained model found! Please train the model first.")

    input_data = np.array([[grid, points, dob]])
    predicted_position = int(model.predict(input_data)[0][0])

    # Get driver name
    driver_info = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_info['forename'].values[0]} {driver_info['surname'].values[0]}" if not driver_info.empty else "Unknown Driver"

    # Get track name
    track_info = races_df[races_df["raceId"] == track_id]
    track_name = track_info["name"].values[0] if not track_info.empty else "Unknown Track"

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_position": predicted_position
    }



