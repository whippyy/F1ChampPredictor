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


def predict_race(*, driver_id: int, circuit_id: str, grid: int):
    try:
        if model is None or scaler is None:
            raise ValueError("Model or scaler not loaded.")

        # Get latest race_id for the circuit in current season
        race_row = races_df[
            (races_df["circuitId"] == circuit_id) & (races_df["year"] == current_year)
        ].sort_values(by="round", ascending=False)

        if race_row.empty:
            raise ValueError(f"No race found at circuit {circuit_id} for {current_year}.")

        race = race_row.iloc[0]
        race_id = race["raceId"]
        year = race["year"]

        # Get constructorId
        result_row = results_df[
            (results_df["raceId"] == race_id) & (results_df["driverId"] == driver_id)
        ]
        if result_row.empty:
            raise ValueError("No race result for this driver.")

        constructor_id = result_row.iloc[0]["constructorId"]

        # Lap time
        avg_lap_time = lap_times_df[
            (lap_times_df["raceId"] == race_id) & (lap_times_df["driverId"] == driver_id)
        ]["milliseconds"].mean()

        # Pit time
        avg_pit_time = pit_stops_df[
            pit_stops_df["raceId"] == race_id
        ]["milliseconds"].mean()

        # Qualifying time
        quali_row = qualifying_df[
            (qualifying_df["raceId"] == race_id) & (qualifying_df["driverId"] == driver_id)
        ]
        if not quali_row.empty:
            q1 = pd.to_numeric(quali_row.iloc[0].get("q1", np.nan), errors="coerce")
            q2 = pd.to_numeric(quali_row.iloc[0].get("q2", np.nan), errors="coerce")
            q3 = pd.to_numeric(quali_row.iloc[0].get("q3", np.nan), errors="coerce")
            avg_quali_time = np.nanmean([q1, q2, q3])
        else:
            avg_quali_time = np.nan

        # Qualifying position
        quali_df_copy = qualifying_df.copy()
        for q in ["q1", "q2", "q3"]:
            quali_df_copy[q] = pd.to_numeric(quali_df_copy[q], errors="coerce")
        quali_df_copy["avg_time"] = quali_df_copy[["q1", "q2", "q3"]].mean(axis=1)
        quali_df_copy["rank"] = quali_df_copy[quali_df_copy["raceId"] == race_id]["avg_time"].rank()
        quali_pos = quali_df_copy[
            (quali_df_copy["raceId"] == race_id) & (quali_df_copy["driverId"] == driver_id)
        ]["rank"].values
        quali_pos = float(quali_pos[0]) if len(quali_pos) > 0 else np.nan

        # Standings
        d_stand = driver_standings_df[
            (driver_standings_df["raceId"] == race_id) & (driver_standings_df["driverId"] == driver_id)
        ]
        driver_points = d_stand["points"].values[0] if not d_stand.empty else 0
        driver_position = d_stand["position"].values[0] if not d_stand.empty else 20

        c_stand = constructor_standings_df[
            (constructor_standings_df["raceId"] == race_id) & (constructor_standings_df["constructorId"] == constructor_id)
        ]
        constructor_points = c_stand["points"].values[0] if not c_stand.empty else 0
        constructor_position = c_stand["position"].values[0] if not c_stand.empty else 10

        # Feature vector
        input_data = pd.DataFrame([{
            "grid": float(grid),
            "avg_lap_time": avg_lap_time,
            "avg_pit_time": avg_pit_time,
            "avg_qualifying_time": avg_quali_time,
            "qualifying_position": quali_pos,
            "driver_points": float(driver_points),
            "driver_position": float(driver_position),
            "constructor_points": float(constructor_points),
            "constructor_position": float(constructor_position)
        }])

        # Fill missing
        input_data.fillna({
            "avg_lap_time": lap_times_df["milliseconds"].median(),
            "avg_pit_time": pit_stops_df["milliseconds"].median(),
            "avg_qualifying_time": 90000,
            "qualifying_position": 10,
            "driver_points": 0,
            "driver_position": 20,
            "constructor_points": 0,
            "constructor_position": 10
        }, inplace=True)

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        predicted_position = int(np.clip(round(prediction * 20), 1, 20))

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
