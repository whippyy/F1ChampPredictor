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
lap_times_df = data["lap_times"]
pit_stops_df = data["pit_stops"]
qualifying_df = data["qualifying"]



# ✅ Compute average lap time per driver per race
avg_lap_time = lap_times_df.groupby(["raceId", "driverId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)

# ✅ Merge it into results_df
results_df = results_df.merge(avg_lap_time, on=["raceId", "driverId"], how="left")

print("✅ Merged avg_lap_time into results_df:", results_df.columns)


# ✅ Ensure all numeric columns are properly converted
numeric_columns = ["milliseconds", "grid", "points", "fastestLapSpeed", "avg_lap_time"]  # Add other relevant columns
for col in numeric_columns:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")  # Convert invalid values to NaN
    results_df[col].fillna(results_df[col].median(), inplace=True)  # Replace NaNs with median values


# ✅ Get current season (2024)
current_season = 2024
current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
current_races = races_df[races_df["year"] == current_season]

# ✅ Get valid drivers & circuits for 2024
valid_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]["driverId"].unique()
valid_circuits = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

def predict_race(driver_id: int, circuit_id: int, grid: int, points: float, fastest_lap: float):
    """
    Predicts the final race result with track-specific data.
    """
    if model is None or scaler is None:
        raise ValueError("No trained model found! Please train the model first.")

    # ✅ Ensure input features match the trained model
    feature_names = ["grid", "race_points", "season_points", "fastestLapSpeed", "avg_lap_time", "avg_pit_time", "avg_qualifying_time"]


    lap_times_df["milliseconds"] = pd.to_numeric(lap_times_df["milliseconds"], errors="coerce")
    lap_times_df["milliseconds"].fillna(lap_times_df["milliseconds"].median(), inplace=True)

    # ✅ Ensure `milliseconds` is numeric (fixes the TypeError issue)
    results_df["milliseconds"] = pd.to_numeric(results_df["milliseconds"], errors="coerce")
    
    # ✅ Compute track-level average lap time
    track_avg_lap_time = results_df[results_df["raceId"].isin(
        races_df[races_df["circuitId"] == circuit_id]["raceId"]
    )]["milliseconds"].mean()

    # ✅ Compute driver-specific average lap time
    driver_avg_lap_time = results_df[
        (results_df["driverId"] == driver_id) & (results_df["raceId"].isin(current_races["raceId"]))
    ]["milliseconds"].mean()

    # ✅ Use driver lap time if available, otherwise fallback to track average
    avg_lap_time = driver_avg_lap_time if not np.isnan(driver_avg_lap_time) else track_avg_lap_time

    # ✅ If still NaN, use the median lap time instead of a hardcoded default
    if np.isnan(avg_lap_time):
        avg_lap_time = results_df["milliseconds"].median()

    print(f"📊 Avg Lap Time for Driver {driver_id} on Track {circuit_id}: {avg_lap_time}")

    # Ensure correct feature names
    results_df.rename(columns={"points": "race_points"}, inplace=True)

    # Fetch race points
    race_points = results_df[results_df["driverId"] == driver_id]["race_points"].max()
    race_points = race_points if not np.isnan(race_points) else 0  # Default to 0 if missing

    # Ensure standings data is loaded
    standings_df = data["driver_standings"]

    # Fetch the latest season points for the driver
    season_points = standings_df[(standings_df["driverId"] == driver_id) & 
                                (standings_df["raceId"].isin(current_races["raceId"]))]["points"].max()
    season_points = season_points if not np.isnan(season_points) else 0  # Default to 0 if missing

    # Fetch average pit stop time for the driver
    # Compute average pit stop time for the driver
    avg_pit_time = pit_stops_df[pit_stops_df["driverId"] == driver_id]["milliseconds"].mean()
    avg_pit_time = avg_pit_time if not np.isnan(avg_pit_time) else pit_stops_df["milliseconds"].median()  # Default to median if missing
    avg_pit_time = avg_pit_time if not np.isnan(avg_pit_time) else results_df["avg_pit_time"].median()  # Default to median

    # Convert qualifying times to numeric values (handling missing values)
    for col in ["q1", "q2", "q3"]:
        qualifying_df[col] = pd.to_numeric(qualifying_df[col], errors="coerce")

    # Compute average qualifying time for the driver
    qualifying_df["avg_qualifying_time"] = qualifying_df[["q1", "q2", "q3"]].mean(axis=1)

    # Fetch the driver's average qualifying time
    avg_qualifying_time = qualifying_df[qualifying_df["driverId"] == driver_id]["avg_qualifying_time"].mean()
    avg_qualifying_time = avg_qualifying_time if not np.isnan(avg_qualifying_time) else qualifying_df["avg_qualifying_time"].median()  # Default to median if missing

    # Fetch the driver's lap times at this circuit
    driver_circuit_laps = lap_times_df[
        (lap_times_df["driverId"] == driver_id) & 
        (lap_times_df["raceId"].isin(races_df[races_df["circuitId"] == circuit_id]["raceId"]))
    ]

    # Get all raceIds for the selected circuit
    circuit_race_ids = races_df[races_df["circuitId"] == circuit_id]["raceId"]

    # Fetch only THIS driver's lap times for THIS circuit
    driver_circuit_laps = lap_times_df[
        (lap_times_df["driverId"] == driver_id) & (lap_times_df["raceId"].isin(circuit_race_ids))
    ]

    # Compute the driver’s average lap time on THIS circuit
    driver_avg_lap_time = driver_circuit_laps["milliseconds"].mean()

    # If no lap data exists, fallback to circuit-wide average lap time
    if np.isnan(driver_avg_lap_time):
        driver_avg_lap_time = lap_times_df[lap_times_df["raceId"].isin(circuit_race_ids)]["milliseconds"].mean()

    # 🔍 Print to verify correct lap times are used
    print(f"🚦 Driver {driver_id} - Avg Lap Time at Circuit {circuit_id}: {driver_avg_lap_time}")


    # Get circuit features
    track_info = circuits_df[circuits_df["circuitId"] == circuit_id][["lat", "lng", "alt"]]
    lat, lng, alt = track_info.iloc[0] if not track_info.empty else (0, 0, 0)

    input_data = pd.DataFrame([[grid, race_points, season_points, fastest_lap, driver_avg_lap_time, 
                            avg_pit_time, avg_qualifying_time]], 
                          columns=feature_names)





    # ✅ Transform input data
    input_data_scaled = scaler.transform(input_data)

    # ✅ Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20

    # ✅ Ensure prediction is valid
    if np.isnan(predicted_position):
        print("⚠️ Warning: Model returned NaN. Assigning default position (10).")
        predicted_position = 10  # Assign a default mid-grid position if NaN

    predicted_position = max(1, min(round(predicted_position), 20))  # Ensure within range


    # ✅ Get driver & track names
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"

    track_row = circuits_df[circuits_df["circuitId"] == circuit_id]
    track_name = track_row["name"].values[0] if not track_row.empty else "Unknown Track"

    return {
        "driver": driver_name,
        "track": track_name,
        "predicted_race_position": predicted_position
    }
