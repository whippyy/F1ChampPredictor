import pandas as pd
import numpy as np
import os
import joblib
from ..data_loader import load_csv_data  # Changed to relative import

MODEL_PATH = os.path.join(os.path.dirname(__file__), "f1_xgb_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load data
data = load_csv_data()
drivers_df = data["drivers"]
circuits_df = data["circuits"]
races_df = data["races"]
results_df = data["results"]
lap_times_df = data["lap_times"]
pit_stops_df = data["pit_stops"]
qualifying_df = data["qualifying"]

# Initialize model and scaler
model = None
scaler = None

# Load model and scaler if files exist
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

def get_driver_stats(driver_id: int, circuit_id: int):
    """Helper function to get driver statistics"""
    stats = {
        "avg_lap_time": lap_times_df[
            (lap_times_df["driverId"] == driver_id) & 
            (lap_times_df["circuitId"] == circuit_id)
        ]["milliseconds"].median(),
        "avg_pit_time": pit_stops_df[
            (pit_stops_df["driverId"] == driver_id) & 
            (pit_stops_df["raceId"].isin(races_df[races_df["circuitId"] == circuit_id]["raceId"]))
        ]["milliseconds"].median(),
        "avg_qualifying_time": qualifying_df[
            (qualifying_df["driverId"] == driver_id) & 
            (qualifying_df["circuitId"] == circuit_id)
        ]["avg_qualifying_time"].median()
    }
    return stats

def predict_race(driver_id: int, circuit_id: int, grid: int):
    """Main prediction function"""
    try:
        # Input validation
        if model is None or scaler is None:
            raise ValueError("Model or scaler not loaded properly")
            
        # Get driver stats
        stats = get_driver_stats(driver_id, circuit_id)
        
        # Prepare input features (must match training exactly)
        input_data = pd.DataFrame([{
            "grid": float(grid),
            "avg_lap_time": stats["avg_lap_time"],
            "avg_pit_time": stats["avg_pit_time"],
            "avg_qualifying_time": stats["avg_qualifying_time"],
            "driver_points": results_df[results_df["driverId"] == driver_id]["points"].sum(),
            "driver_position": results_df[results_df["driverId"] == driver_id]["positionOrder"].mean(),
            "constructor_points": results_df[results_df["driverId"] == driver_id]["constructorId"].map(
                lambda x: results_df[results_df["constructorId"] == x]["points"].sum()).mean(),
            "constructor_position": results_df[results_df["driverId"] == driver_id]["constructorId"].map(
                lambda x: results_df[results_df["constructorId"] == x]["positionOrder"].mean()).mean()
        }])
        
        # Fill any remaining NaN values
        input_data = input_data.fillna({
            "avg_lap_time": lap_times_df["milliseconds"].median(),
            "avg_pit_time": pit_stops_df["milliseconds"].median(),
            "avg_qualifying_time": qualifying_df["avg_qualifying_time"].median(),
            "driver_points": 0,
            "driver_position": 20,
            "constructor_points": 0,
            "constructor_position": 10
        })
        
        # Scale features
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        predicted_position = max(1, min(round(prediction * 20), 20))
        
        # Prepare result - MATCH WHAT predictions.py EXPECTS
        driver_info = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
        driver_name = f"{driver_info['forename']} {driver_info['surname']}"
        circuit_info = circuits_df[circuits_df["circuitId"] == circuit_id].iloc[0]
        
        return {
            "driver_id": driver_id,
            "driver_name": driver_name,
            "predicted_race_position": predicted_position,
            "track": circuit_info["name"],
            "status": "success"
        }

        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }