import pandas as pd
import numpy as np
import os
import joblib
from ..data_loader import load_csv_data
from app.data_loader import f1_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "f1_xgb_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load data
data = load_csv_data()
drivers_df = f1_data.data["drivers"]
circuits_df = f1_data.data["circuits"]
races_df = f1_data.data["races"]
results_df = f1_data.data["results"]
lap_times_df = f1_data.data["lap_times"]
pit_stops_df = f1_data.data["pit_stops"]
qualifying_df = f1_data.data["qualifying"]
driver_standings_df = f1_data.data["driver_standings"]
constructor_standings_df = f1_data.data["standings"]
constructors_df = f1_data.data["constructors"]


# Load model and scaler
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

current_year = 2024

def get_track_features(circuit_id):
    """Extract track-specific characteristics"""
    circuit = circuits_df[circuits_df["circuitId"] == circuit_id].iloc[0]
    
    # Get historical data for this circuit
    circuit_races = races_df[races_df["circuitId"] == circuit_id]
    circuit_results = results_df[results_df["raceId"].isin(circuit_races["raceId"])]
    
    # Calculate track-specific stats
    if not circuit_results.empty:
        avg_pit_stops = pit_stops_df[pit_stops_df["raceId"].isin(circuit_races["raceId"])]["stop"].mean()
        avg_lap_time = lap_times_df[lap_times_df["raceId"].isin(circuit_races["raceId"])]["milliseconds"].mean()
        overtaking_factor = circuit_results.groupby("grid")["position"].mean().diff().mean()  # Avg position change
    else:
        avg_pit_stops = 2.0
        avg_lap_time = 90000
        overtaking_factor = 0.5
    
    return {
        "circuit_length": circuit.get("length", 5000),
        "circuit_corners": circuit.get("corners", 12),
        "circuit_altitude": circuit.get("altitude", 200),
        "circuit_avg_pit_stops": avg_pit_stops,
        "circuit_avg_lap_time": avg_lap_time,
        "circuit_overtaking_factor": overtaking_factor
    }

def get_driver_track_history(driver_id, circuit_id):
    """Get driver's historical performance at this track"""
    circuit_races = races_df[races_df["circuitId"] == circuit_id]
    driver_results = results_df[
        (results_df["driverId"] == driver_id) & 
        (results_df["raceId"].isin(circuit_races["raceId"]))
    ]
    
    if not driver_results.empty:
        avg_finish = driver_results["positionOrder"].mean()
        best_finish = driver_results["positionOrder"].min()
        finish_rate = len(driver_results[driver_results["positionOrder"] <= 10]) / len(driver_results)
    else:
        avg_finish = 15
        best_finish = 20
        finish_rate = 0.2
    
    return {
        "driver_avg_finish": avg_finish,
        "driver_best_finish": best_finish,
        "driver_finish_rate": finish_rate
    }

def predict_race(*, driver_id: int, circuit_id: str, grid: int):
    try:
        if model is None or scaler is None:
            raise ValueError("Model or scaler not loaded.")

        # Get driver's history at this track
        driver_history = get_driver_track_history(driver_id, circuit_id)
        
        # Get latest standings
        latest_race = races_df[races_df["year"] == current_year].sort_values("round").iloc[-1]
        latest_race_id = latest_race["raceId"]
        
        # Get driver and constructor standings
        driver_standing = driver_standings_df[
            (driver_standings_df["raceId"] == latest_race_id) &
            (driver_standings_df["driverId"] == driver_id)
        ].iloc[0] if not driver_standings_df.empty else None
        
        constructor_id = results_df[
            (results_df["driverId"] == driver_id) &
            (results_df["raceId"] == latest_race_id)
        ]["constructorId"].values[0]
        
        constructor_standing = constructor_standings_df[
            (constructor_standings_df["raceId"] == latest_race_id) &
            (constructor_standings_df["constructorId"] == constructor_id)
        ].iloc[0] if not constructor_standings_df.empty else None

        # Prepare input features matching the trained model
        input_data = pd.DataFrame([{
            "grid": float(grid),
            "quali_percentile": 0.5,  # Placeholder - calculate from qualifying data
            "driver_circuit_races": driver_history.get("driver_circuit_races", 0),
            "driver_circuit_avg_finish": driver_history.get("driver_avg_finish", 15),
            "driver_circuit_best_finish": driver_history.get("driver_best_finish", 20),
            "driver_circuit_top3_rate": driver_history.get("driver_finish_rate", 0),
            "recent_avg_finish": 10,  # Should calculate from last 5 races
            "recent_avg_points": 5,   # Should calculate from last 5 races
            "current_points": driver_standing["points"] if driver_standing else 0,
            "current_standing": driver_standing["position"] if driver_standing else 20,
            "constructor_points": constructor_standing["points"] if constructor_standing else 0,
            "constructor_standing": constructor_standing["position"] if constructor_standing else 10
        }])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_position = int(np.clip(round(prediction), 1, 20))

        return {
            "status": "success",
            "predicted_race_position": predicted_position,
            "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0],
            "driver_id": driver_id
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }