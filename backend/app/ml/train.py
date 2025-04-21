import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.data_loader import load_csv_data
import joblib
import os
from xgboost import plot_importance
import matplotlib.pyplot as plt

print("🟢 Training script started") 

# Create ml directory if it doesn't exist
os.makedirs("app/ml", exist_ok=True)

def validate_data(data):
    """Validate that all required data is present"""
    required_tables = {
        'drivers': ['driverId'],
        'results': ['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder'],
        'races': ['raceId', 'circuitId', 'year', 'round'],
        'circuits': ['circuitId', 'name'],
        'lap_times': ['raceId', 'driverId', 'milliseconds'],
        'pit_stops': ['raceId', 'driverId', 'milliseconds'],
        'qualifying': ['raceId', 'driverId', 'q1', 'q2', 'q3'],
        'driver_standings': ['raceId', 'driverId', 'points', 'position'],
        'standings': ['raceId', 'constructorId', 'points', 'position'],
        'constructors': ['constructorId']
    }
    
    for table, cols in required_tables.items():
        if table not in data:
            raise ValueError(f"Missing required table: {table}")
        missing = set(cols) - set(data[table].columns)
        if missing:
            raise ValueError(f"Table {table} missing columns: {missing}")
        
def prepare_features(data):
    """Prepare features with track-specific characteristics"""
    print("🟡 Preparing features...")
    
    # Data inspection
    print(f"🔹 Results shape: {data['results'].shape}")
    print(f"🔹 Circuits shape: {data['circuits'].shape}")
    print(f"🔹 Circuits columns: {data['circuits'].columns.tolist()}")
    print("\n=== DATA SNAPSHOT ===")
    print("Circuits sample:")
    print(data['circuits'][['circuitId', 'name']].head(3))
    
    # Load all data files
    drivers = data["drivers"]
    results = data["results"]
    races = data["races"]
    circuits = data["circuits"]
    lap_times = data["lap_times"]
    pit_stops = data["pit_stops"]
    qualifying = data["qualifying"]
    driver_standings = data["driver_standings"].copy()
    constructor_standings = data["standings"].copy()
    constructors = data["constructors"]

    # Verify required circuit columns exist
    circuit_columns = circuits.columns
    required_circuit_cols = {'circuitId', 'name'}
    missing_cols = required_circuit_cols - set(circuit_columns)
    
    if missing_cols:
        raise ValueError(f"Circuits DataFrame missing required columns: {missing_cols}")

    # Add default values for optional circuit characteristics
    if 'length' not in circuits.columns:
        print("⚠️ 'length' column not found in circuits, using default values")
        circuits['length'] = 5000
    if 'corners' not in circuits.columns:
        print("⚠️ 'corners' column not found in circuits, using default values")
        circuits['corners'] = 12
    if 'altitude' not in circuits.columns:
        circuits['altitude'] = 200

    # First merge races with circuits to get circuitId for lap times
    races_with_circuits = races[["raceId", "circuitId"]].merge(
        circuits[["circuitId", "length", "corners", "altitude"]],
        on="circuitId",
        how="left"
    )

    # Calculate track average lap times - first join lap times with race circuit info
    lap_times_with_circuit = lap_times.merge(
        races_with_circuits,
        on="raceId",
        how="left"
    )
    
    # Now calculate track stats
    track_lap_stats = lap_times_with_circuit.groupby("circuitId")["milliseconds"].agg(["mean", "std"]).reset_index()
    track_lap_stats.columns = ["circuitId", "track_avg_lap", "track_std_lap"]
    
    # Similarly for pit stops
    pit_stops_with_circuit = pit_stops.merge(
        races_with_circuits,
        on="raceId",
        how="left"
    )
    track_pit_stats = pit_stops_with_circuit.groupby("circuitId")["milliseconds"].mean().reset_index()
    track_pit_stats.columns = ["circuitId", "track_avg_pit"]
    
    # Merge base data
    merged_df = results.merge(races_with_circuits, on="raceId", how="left")
    merged_df = merged_df.merge(drivers, on="driverId", how="left")
    merged_df = merged_df.merge(constructors, on="constructorId", how="left")

    # Calculate normalized track characteristics
    merged_df["circuit_length_norm"] = merged_df["length"] / merged_df["length"].max()
    merged_df["circuit_corners_norm"] = merged_df["corners"] / merged_df["corners"].max()
    
    # Track average performance metrics
    track_lap_stats = lap_times.groupby("circuitId")["milliseconds"].agg(["mean", "std"]).reset_index()
    track_lap_stats.columns = ["circuitId", "track_avg_lap", "track_std_lap"]
    merged_df = merged_df.merge(track_lap_stats, on="circuitId", how="left")
    
    track_pit_stats = pit_stops.merge(races[["raceId", "circuitId"]], on="raceId").groupby("circuitId")["milliseconds"].mean().reset_index()
    track_pit_stats.columns = ["circuitId", "track_avg_pit"]
    merged_df = merged_df.merge(track_pit_stats, on="circuitId", how="left")
    
    # Driver's historical performance at this track
    driver_track_history = results.merge(races[["raceId", "circuitId"]], on="raceId") \
        .groupby(["driverId", "circuitId"])["positionOrder"].agg(["mean", "min", "count"]).reset_index()
    driver_track_history.columns = ["driverId", "circuitId", "driver_track_avg", "driver_track_best", "driver_track_races"]
    merged_df = merged_df.merge(driver_track_history, on=["driverId", "circuitId"], how="left")
    
    # Calculate relative performance metrics
    merged_df["lap_time_ratio"] = merged_df.groupby(["raceId", "driverId"])["milliseconds"].transform("mean") / merged_df["track_avg_lap"]
    merged_df["pit_time_ratio"] = merged_df.groupby(["raceId", "driverId"])["milliseconds"].transform("mean") / merged_df["track_avg_pit"]
    
    # Qualifying performance
    qualifying = qualifying.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
    for col in ["q1", "q2", "q3"]:
        qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")
    qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
    qualifying = qualifying.merge(track_lap_stats, on="circuitId", how="left")
    qualifying["quali_time_ratio"] = qualifying["avg_qualifying_time"] / qualifying["track_avg_lap"]
    qualifying_pos = qualifying[["raceId", "driverId", "quali_time_ratio"]]
    merged_df = merged_df.merge(qualifying_pos, on=["raceId", "driverId"], how="left")

    # Standings data
    driver_standings = driver_standings[["raceId", "driverId", "points", "position"]].copy()
    driver_standings.rename(columns={"points": "driver_points", "position": "driver_position"}, inplace=True)
    merged_df = merged_df.merge(driver_standings, on=["raceId", "driverId"], how="left")

    constructor_standings = constructor_standings[["raceId", "constructorId", "points", "position"]].copy()
    constructor_standings.rename(columns={"points": "constructor_points", "position": "constructor_position"}, inplace=True)
    merged_df = merged_df.merge(constructor_standings, on=["raceId", "constructorId"], how="left")

    # Handle missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan)
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        if col.endswith("_ratio"):
            merged_df[col] = merged_df[col].fillna(1.0)
        elif "track" in col:
            if "lap" in col:
                merged_df[col] = merged_df[col].fillna(merged_df["milliseconds"].median())
            elif "pit" in col:
                merged_df[col] = merged_df[col].fillna(pit_stops["milliseconds"].median())
        elif "driver_track" in col:
            if "avg" in col:
                merged_df[col] = merged_df[col].fillna(10)
            elif "best" in col:
                merged_df[col] = merged_df[col].fillna(20)
            elif "races" in col:
                merged_df[col] = merged_df[col].fillna(0)
    
    # Clean up
    del drivers, results, races, circuits, lap_times, pit_stops, qualifying
    del driver_standings, constructor_standings, constructors
    
    return merged_df

def train_models():
    print("🟡 Starting model training...")
    data = load_csv_data()
    validate_data(data)
    df = prepare_features(data)
    
    # Enhanced feature set
    race_features = [
        "grid",
        "lap_time_ratio",
        "pit_time_ratio",
        "quali_time_ratio",
        "driver_points",
        "driver_position",
        "constructor_points",
        "constructor_position",
        "circuit_length_norm",
        "circuit_corners_norm",
        "driver_track_avg",
        "driver_track_best"
    ]
    
    X = df[race_features]
    y = df["positionOrder"] / df["positionOrder"].max()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mae'
    )
    
    model.fit(
        X_train_scaled, 
        y_train, 
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 Enhanced Race Model Performance:")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE: {mae:.4f}")
    
    # Save artifacts
    joblib.dump(scaler, "app/ml/scaler.pkl")
    joblib.dump(model, "app/ml/f1_xgb_model.pkl")
    
    # Feature importance visualization
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=15)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("app/ml/feature_importance.png")
    print("📊 Feature importance plot saved to app/ml/feature_importance.png")
    
    print("\n✅ Enhanced model trained and saved successfully!")

if __name__ == "__main__":
    print("🔴 Main block executing")
    train_models()
    print("✅ Training completed successfully")