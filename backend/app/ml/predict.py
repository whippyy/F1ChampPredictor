import pandas as pd
import numpy as np
import os
import joblib
from ..data_loader import load_csv_data

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
driver_standings_df = data["driver_standings"]
constructor_standings_df = data["standings"]
constructors_df = data["constructors"]

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

        # Get track features
        track_features = get_track_features(circuit_id)
        
        # Get driver's history at this track
        driver_history = get_driver_track_history(driver_id, circuit_id)
        
        # Get latest race_id for the circuit in current season
        race_row = races_df[
            (races_df["circuitId"] == circuit_id) & 
            (races_df["year"] == current_year)
        ].sort_values(by="round", ascending=False)

        if race_row.empty:
            # If no race this year, use most recent race at this circuit
            race_row = races_df[races_df["circuitId"] == circuit_id].sort_values(
                by=["year", "round"], ascending=[False, False]
            )
            if race_row.empty:
                raise ValueError(f"No race found at circuit {circuit_id}.")
            year = race_row.iloc[0]["year"]
        else:
            year = current_year

        race = race_row.iloc[0]
        race_id = race["raceId"]

        # Get constructorId
        result_row = results_df[
            (results_df["raceId"] == race_id) & 
            (results_df["driverId"] == driver_id)
        ]
        if result_row.empty:
            # If no result for this race, use most recent constructor
            constructor_row = results_df[
                (results_df["driverId"] == driver_id) & 
                (results_df["raceId"].isin(races_df[races_df["year"] == year]["raceId"]))
            ].sort_values(by="raceId", ascending=False)
            if constructor_row.empty:
                raise ValueError("No constructor data for this driver.")
            constructor_id = constructor_row.iloc[0]["constructorId"]
        else:
            constructor_id = result_row.iloc[0]["constructorId"]

        # Lap time (relative to track average)
        driver_lap_time = lap_times_df[
            (lap_times_df["raceId"] == race_id) & 
            (lap_times_df["driverId"] == driver_id)
        ]["milliseconds"].mean()
        lap_time_ratio = driver_lap_time / track_features["circuit_avg_lap_time"] if not np.isnan(driver_lap_time) else 1.0

        # Pit time (relative to track average)
        driver_pit_time = pit_stops_df[
            (pit_stops_df["raceId"] == race_id) & 
            (pit_stops_df["driverId"] == driver_id)
        ]["milliseconds"].mean()
        pit_time_ratio = driver_pit_time / pit_stops_df["milliseconds"].median() if not np.isnan(driver_pit_time) else 1.0

        # Qualifying performance
        quali_row = qualifying_df[
            (qualifying_df["raceId"] == race_id) & 
            (qualifying_df["driverId"] == driver_id)
        ]
        if not quali_row.empty:
            q1 = pd.to_numeric(quali_row.iloc[0].get("q1", np.nan), errors="coerce")
            q2 = pd.to_numeric(quali_row.iloc[0].get("q2", np.nan), errors="coerce")
            q3 = pd.to_numeric(quali_row.iloc[0].get("q3", np.nan), errors="coerce")
            quali_times = [q for q in [q1, q2, q3] if not np.isnan(q)]
            avg_quali_time = np.mean(quali_times) if quali_times else np.nan
            quali_time_ratio = avg_quali_time / track_features["circuit_avg_lap_time"] if not np.isnan(avg_quali_time) else 1.0
        else:
            quali_time_ratio = 1.0
            avg_quali_time = np.nan

        # Qualifying position
        quali_df_copy = qualifying_df.copy()
        for q in ["q1", "q2", "q3"]:
            quali_df_copy[q] = pd.to_numeric(quali_df_copy[q], errors="coerce")
        quali_df_copy["avg_time"] = quali_df_copy[["q1", "q2", "q3"]].mean(axis=1)
        quali_df_copy["rank"] = quali_df_copy[quali_df_copy["raceId"] == race_id]["avg_time"].rank()
        quali_pos = quali_df_copy[
            (quali_df_copy["raceId"] == race_id) & 
            (quali_df_copy["driverId"] == driver_id)
        ]["rank"].values
        quali_pos = float(quali_pos[0]) if len(quali_pos) > 0 else 10.0

        # Standings
        latest_race = races_df[races_df["year"] == year].sort_values(by="round", ascending=False).iloc[0]
        latest_race_id = latest_race["raceId"]
        
        d_stand = driver_standings_df[
            (driver_standings_df["raceId"] == latest_race_id) & 
            (driver_standings_df["driverId"] == driver_id)
        ]
        driver_points = d_stand["points"].values[0] if not d_stand.empty else 0
        driver_position = d_stand["position"].values[0] if not d_stand.empty else 20

        c_stand = constructor_standings_df[
            (constructor_standings_df["raceId"] == latest_race_id) & 
            (constructor_standings_df["constructorId"] == constructor_id)
        ]
        constructor_points = c_stand["points"].values[0] if not c_stand.empty else 0
        constructor_position = c_stand["position"].values[0] if not c_stand.empty else 10

        # Feature vector with track-specific features
        input_data = pd.DataFrame([{
            "grid": float(grid),
            "lap_time_ratio": lap_time_ratio,
            "pit_time_ratio": pit_time_ratio,
            "quali_time_ratio": quali_time_ratio,
            "qualifying_position": quali_pos,
            "driver_points": float(driver_points),
            "driver_position": float(driver_position),
            "constructor_points": float(constructor_points),
            "constructor_position": float(constructor_position),
            "circuit_length": track_features["circuit_length"],
            "circuit_corners": track_features["circuit_corners"],
            "circuit_altitude": track_features["circuit_altitude"],
            "circuit_avg_pit_stops": track_features["circuit_avg_pit_stops"],
            "circuit_overtaking_factor": track_features["circuit_overtaking_factor"],
            "driver_avg_finish": driver_history["driver_avg_finish"],
            "driver_best_finish": driver_history["driver_best_finish"],
            "driver_finish_rate": driver_history["driver_finish_rate"]
        }])

        # Fill missing values with reasonable defaults
        input_data.fillna({
            "lap_time_ratio": 1.0,
            "pit_time_ratio": 1.0,
            "quali_time_ratio": 1.0,
            "qualifying_position": 10.0,
            "driver_points": 0.0,
            "driver_position": 20.0,
            "constructor_points": 0.0,
            "constructor_position": 10.0,
            "circuit_length": 5000,
            "circuit_corners": 12,
            "circuit_altitude": 200,
            "circuit_avg_pit_stops": 2.0,
            "circuit_overtaking_factor": 0.5,
            "driver_avg_finish": 15.0,
            "driver_best_finish": 20.0,
            "driver_finish_rate": 0.2
        }, inplace=True)

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_position = int(np.clip(round(prediction), 1, 20))

        return {
            "status": "success",
            "predicted_race_position": predicted_position,
            "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0],
            "driver_id": driver_id,
            "track_features": {
                "length": track_features["circuit_length"],
                "corners": track_features["circuit_corners"],
                "overtaking_factor": track_features["circuit_overtaking_factor"]
            },
            "driver_history": driver_history
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }