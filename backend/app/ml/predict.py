import pandas as pd
import numpy as np
import os
import joblib
from ..data_loader import load_csv_data  # Relative import

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

# Initialize model and scaler
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


def predict_race(driver_id: int, race_id: int, grid: int):
    try:
        if model is None or scaler is None:
            raise ValueError("Model or scaler not loaded.")

        # Get race and circuit info
        race = races_df[races_df["raceId"] == race_id].iloc[0]
        circuit_id = race["circuitId"]
        year = race["year"]

        # Get constructorId for this driver in this race
        result_row = results_df[
            (results_df["raceId"] == race_id) & (results_df["driverId"] == driver_id)
        ]
        if result_row.empty:
            raise ValueError("No result found for this driver in the specified race.")

        constructor_id = result_row.iloc[0]["constructorId"]

        # Avg lap time at this circuit
        avg_lap_time = lap_times_df[
            (lap_times_df["driverId"] == driver_id) & (lap_times_df["raceId"] == race_id)
        ]["milliseconds"].mean()

        # Avg pit stop time in this race
        avg_pit_time = pit_stops_df[
            pit_stops_df["raceId"] == race_id
        ]["milliseconds"].mean()

        # Avg qualifying time
        quali_row = qualifying_df[
            (qualifying_df["driverId"] == driver_id) & (qualifying_df["raceId"] == race_id)
        ]
        if not quali_row.empty:
            q1 = pd.to_numeric(quali_row.iloc[0].get("q1", np.nan), errors="coerce")
            q2 = pd.to_numeric(quali_row.iloc[0].get("q2", np.nan), errors="coerce")
            q3 = pd.to_numeric(quali_row.iloc[0].get("q3", np.nan), errors="coerce")
            avg_qualifying_time = np.nanmean([q1, q2, q3])
        else:
            avg_qualifying_time = np.nan

        # Qualifying position
        qualifying_df_copy = qualifying_df.copy()
        for col in ["q1", "q2", "q3"]:
            qualifying_df_copy[col] = pd.to_numeric(qualifying_df_copy[col], errors="coerce")
        qualifying_df_copy["avg_qualifying_time"] = qualifying_df_copy[["q1", "q2", "q3"]].mean(axis=1)
        qualifying_df_copy["qualifying_position"] = qualifying_df_copy[
            qualifying_df_copy["raceId"] == race_id
        ]["avg_qualifying_time"].rank()

        qualifying_position = qualifying_df_copy[
            (qualifying_df_copy["raceId"] == race_id) & (qualifying_df_copy["driverId"] == driver_id)
        ]["qualifying_position"].values
        qualifying_position = float(qualifying_position[0]) if len(qualifying_position) > 0 else np.nan

        # Standings
        driver_stand = driver_standings_df[
            (driver_standings_df["raceId"] == race_id) & (driver_standings_df["driverId"] == driver_id)
        ]
        driver_points = driver_stand["points"].values[0] if not driver_stand.empty else 0
        driver_position = driver_stand["position"].values[0] if not driver_stand.empty else 20

        constructor_stand = constructor_standings_df[
            (constructor_standings_df["raceId"] == race_id) & (constructor_standings_df["constructorId"] == constructor_id)
        ]
        constructor_points = constructor_stand["points"].values[0] if not constructor_stand.empty else 0
        constructor_position = constructor_stand["position"].values[0] if not constructor_stand.empty else 10

        # Build feature vector
        input_data = pd.DataFrame([{
            "grid": float(grid),
            "avg_lap_time": avg_lap_time,
            "avg_pit_time": avg_pit_time,
            "avg_qualifying_time": avg_qualifying_time,
            "driver_points": float(driver_points),
            "driver_position": float(driver_position),
            "constructor_points": float(constructor_points),
            "constructor_position": float(constructor_position),
            "qualifying_position": qualifying_position
        }])

        # Fill NaNs
        input_data.fillna({
            "avg_lap_time": lap_times_df["milliseconds"].median(),
            "avg_pit_time": pit_stops_df["milliseconds"].median(),
            "avg_qualifying_time": qualifying_df["avg_qualifying_time"].median() if "avg_qualifying_time" in qualifying_df else 90000,
            "qualifying_position": 10,
            "driver_points": 0,
            "driver_position": 20,
            "constructor_points": 0,
            "constructor_position": 10
        }, inplace=True)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        predicted_position = max(1, min(round(prediction * 20), 20))

        # Info
        driver_info = drivers_df[drivers_df["driverId"] == driver_id].iloc[0]
        circuit_info = circuits_df[circuits_df["circuitId"] == circuit_id].iloc[0]

        return {
            "driver_id": driver_id,
            "driver_name": f"{driver_info['forename']} {driver_info['surname']}",
            "predicted_race_position": predicted_position,
            "track": circuit_info["name"],
            "year": year,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }
