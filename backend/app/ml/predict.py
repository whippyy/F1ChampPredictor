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

# Ensure circuitId exists in lap_times_df before merging
if "circuitId" not in lap_times_df.columns:
    lap_times_df = lap_times_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Compute driver’s average lap time per circuit
avg_lap_time = lap_times_df.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)

# Ensure circuitId exists in results_df before merging
if "circuitId" not in results_df.columns:
    results_df = results_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Merge avg_lap_time into results_df
results_df = results_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

# Ensure circuitId exists in qualifying_df before merging
if "circuitId" not in qualifying_df.columns:
    qualifying_df = qualifying_df.merge(races_df[["raceId", "circuitId"]], on="raceId", how="left")

# Convert qualifying times (q1, q2, q3) to numeric values (in milliseconds)
for col in ["q1", "q2", "q3"]:
    qualifying_df[col] = pd.to_numeric(qualifying_df[col], errors="coerce")

# Calculate average qualifying time
qualifying_df["avg_qualifying_time"] = qualifying_df[["q1", "q2", "q3"]].mean(axis=1)

# Ensure numeric columns are properly converted
numeric_columns = ["grid", "fastestLapSpeed", "avg_lap_time"]
for col in numeric_columns:
    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
    results_df[col] = results_df[col].fillna(results_df[col].median())

# In your predict.py, add these debug prints BEFORE the prediction:
print("\n=== DEBUGGING INPUT DATA ===")
print("Driver ID:", driver_id)
print("Circuit ID:", circuit_id)
print("Raw input features:")
print(input_data)

# Add validation before prediction
if input_data.isnull().values.any():
    print("WARNING: NaN values detected in features:")
    print(input_data.isnull().sum())
    input_data = input_data.fillna(input_data.median())

print("Final features being sent to model:")
print(input_data)

def predict_race(driver_id: int, circuit_id: int, grid: int):
    """
    Predicts the final race result using all the training features.
    """
    
    # 1. Verify model and scaler loaded properly
    if model is None:
        raise ValueError("Model failed to load!")
    if scaler is None:
        raise ValueError("Scaler failed to load!")

    # 2. Verify feature names match exactly
    required_features = [
        "grid", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
        "driver_points", "driver_position", "constructor_points", "constructor_position"
    ]
    assert all(feat in input_data.columns for feat in required_features), \
        f"Missing features! Expected: {required_features}"

    # Fetch driver info
    driver_row = drivers_df[drivers_df["driverId"] == driver_id]
    driver_name = f"{driver_row['forename'].values[0]} {driver_row['surname'].values[0]}" if not driver_row.empty else "Unknown Driver"
    driver_code = driver_row["code"].values[0] if "code" in driver_row.columns and not driver_row.empty else None

    # Fetch team info
    team_id = results_df[results_df["driverId"] == driver_id]["constructorId"].values[0] if "constructorId" in results_df.columns else None
    team_name, team_code = "Unknown Team", None
    if team_id is not None:
        team_row = data["constructors"][data["constructors"]["constructorId"] == team_id]
        if not team_row.empty:
            team_name = team_row["name"].values[0]
            team_code = team_row["constructorRef"].values[0] if "constructorRef" in team_row.columns else None

    # Extract all features used for training
    feature_columns = [
        "grid", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
        "driver_points", "driver_position", "constructor_points", "constructor_position"
    ]

    # Fetch race-specific features
    driver_results = results_df[(results_df["driverId"] == driver_id) & (results_df["circuitId"] == circuit_id)]

    # Compute missing features
    avg_lap_time = driver_results["avg_lap_time"].mean()
    avg_pit_time = pit_stops_df[(pit_stops_df["driverId"] == driver_id) & 
                                (pit_stops_df["raceId"].isin(races_df[races_df["circuitId"] == circuit_id]["raceId"]))]["milliseconds"].mean()
    avg_qualifying_time = qualifying_df[(qualifying_df["driverId"] == driver_id) & 
                                        (qualifying_df["circuitId"] == circuit_id)]["avg_qualifying_time"].mean()
    fastest_lap_speed = driver_results["fastestLapSpeed"].mean()
    wins = driver_results["positionOrder"].eq(1).sum()  # Number of wins
    points = driver_results["points"].sum()
    
    constructor_standing = driver_results["constructorId"].map(lambda x: results_df[results_df["constructorId"] == x]["points"].sum()).mean()
    driver_standing = points  # Could be replaced with a ranking if available

    # Fill missing values with dataset medians
    avg_lap_time = avg_lap_time if not np.isnan(avg_lap_time) else results_df["avg_lap_time"].median()
    avg_pit_time = avg_pit_time if not np.isnan(avg_pit_time) else pit_stops_df["milliseconds"].median()
    avg_qualifying_time = avg_qualifying_time if not np.isnan(avg_qualifying_time) else qualifying_df["avg_qualifying_time"].median()
    fastest_lap_speed = fastest_lap_speed if not np.isnan(fastest_lap_speed) else results_df["fastestLapSpeed"].median()
    wins = wins if wins is not None else 0
    points = points if not np.isnan(points) else 0
    constructor_standing = constructor_standing if not np.isnan(constructor_standing) else results_df["points"].median()
    driver_standing = driver_standing if not np.isnan(driver_standing) else 0

    # Construct feature vector
    input_data = pd.DataFrame([[
        grid, 
        avg_lap_time, 
        avg_pit_time, 
        avg_qualifying_time,
        points,  # driver_points
        driver_standing,  # driver_position
        constructor_standing,  # constructor_points
        constructor_standing  # constructor_position (you may need to calculate this separately)
    ]], columns=feature_columns)
    
    if input_data.isnull().values.any():
        print("⚠️ Warning: NaN values detected in input features:")
        print(input_data.isnull().sum())
        # Fill NaN with median values from training
        input_data = input_data.fillna({
            "grid": results_df["grid"].median(),
            "avg_lap_time": results_df["avg_lap_time"].median(),
            # ... fill other features similarly
        })

    # Verify again before prediction
    if input_data.isnull().values.any():
        raise ValueError("NaN values still present after filling!")
    
    # Normalize input data
    input_data_scaled = scaler.transform(input_data)

    # Predict race position
    predicted_position = model.predict(input_data_scaled)[0][0] * 20
    
    # Add validation
    if np.isnan(predicted_position):
        print("⚠️ Model predicted NaN! Using median position as fallback")
        predicted_position = 10  # Middle position as fallback
    else:
        predicted_position = max(1, min(round(predicted_position), 20))
                                
    # Final result
    result = {
        "driver_id": driver_id,
        "driver": driver_name,
        "driver_code": driver_code,
        "team": team_name,
        "team_code": team_code,
        "track": circuits_df[circuits_df["circuitId"] == circuit_id]["name"].values[0] if circuit_id in circuits_df["circuitId"].values else "Unknown Track",
        "predicted_race_position": predicted_position
    }

    return result
