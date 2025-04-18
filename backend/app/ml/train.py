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

# Create ml directory if it doesn't exist
os.makedirs("app/ml", exist_ok=True)

def prepare_features(data):
    """Prepare features for both qualifying and race predictions"""
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

    # Merge base data
    merged_df = results.merge(races[["raceId", "circuitId", "year"]], on="raceId", how="left")
    merged_df = merged_df.merge(drivers, on="driverId", how="left")
    merged_df = merged_df.merge(circuits, on="circuitId", how="left")
    merged_df = merged_df.merge(constructors, on="constructorId", how="left")

    # Calculate time-based features
    # Pit stop times
    pit_stop_avg = pit_stops.groupby("raceId")["milliseconds"].mean().reset_index()
    pit_stop_avg.rename(columns={"milliseconds": "avg_pit_time"}, inplace=True)
    merged_df = merged_df.merge(pit_stop_avg, on="raceId", how="left")

    # Lap times
    lap_times = lap_times.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
    avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
    avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
    merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

    # Qualifying times
    qualifying = qualifying.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
    for col in ["q1", "q2", "q3"]:
        qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")
    qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
    qualifying_avg = qualifying.groupby(["driverId", "circuitId"])["avg_qualifying_time"].mean().reset_index()
    merged_df = merged_df.merge(qualifying_avg, on=["driverId", "circuitId"], how="left")
    
    # Add qualifying position as a feature for race predictions
    qualifying["qualifying_position"] = qualifying.groupby("raceId")["avg_qualifying_time"].rank()
    qualifying_pos = qualifying[["raceId", "driverId", "qualifying_position"]]
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
        # Replace inf/-inf with NaN first
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], np.nan)
        # Fill remaining NaN with column median
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        # If still NaN (all values were NaN), fill with 0
        merged_df[col] = merged_df[col].fillna(0)
    
    return merged_df

def train_models():
    # Load and prepare data
    data = load_csv_data()

    print("\n=== DATA QUALITY REPORT ===")
    for name, df in data.items():
        print(f"\n{name}:")
        print(f"Shape: {df.shape}")
        print(f"NaN count: {df.isna().sum().sum()}")
        print(f"Inf count: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")

    df = prepare_features(data)
    
    # Common features for both models
    base_features = [
        "grid", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
        "driver_points", "driver_position", "constructor_points", "constructor_position"
    ]
    
    # Train qualifying position model
    print("\n=== TRAINING QUALIFYING MODEL ===")
    qual_features = base_features.copy()
    X_qual = df[qual_features]
    y_qual = df["qualifying_position"] / df["qualifying_position"].max()
    
    # Check for invalid values in labels
    if y_qual.isnull().any() or np.isinf(y_qual).any():
        print("Found invalid values in qualifying labels. Fixing...")
        y_qual = y_qual.fillna(y_qual.median())
        y_qual = y_qual.replace([np.inf, -np.inf], y_qual.median())
    
    # Train race position model
    print("\n=== TRAINING RACE POSITION MODEL ===")
    race_features = base_features + ["qualifying_position"]
    X_race = df[race_features]
    y_race = df["positionOrder"] / df["positionOrder"].max()
    
    # Check for invalid values in labels
    if y_race.isnull().any() or np.isinf(y_race).any():
        print("Found invalid values in race labels. Fixing...")
        y_race = y_race.fillna(y_race.median())
        y_race = y_race.replace([np.inf, -np.inf], y_race.median())
    
    for model_type, X, y in [("qual", X_qual, y_qual), ("race", X_race, y_race)]:
        print(f"\nTraining {model_type} model...")
        
        # Validate labels
        print(f"Label stats before cleaning: min={y.min()}, max={y.max()}, NaN={y.isna().sum()}, Inf={(y == np.inf).sum()}")
        
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median())
        
        # If all values are bad, use default (0.5 for normalized target)
        if len(y.unique()) == 1 or y.isna().any():
            y = pd.Series([0.5]*len(y), index=y.index)
            print("⚠️ Used fallback labels due to data issues")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,  # Reduced for faster testing
            learning_rate=0.05,
            max_depth=4,  # Reduced for faster testing
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=True)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n📊 {model_type.upper()} Model Performance:")
        print(f"🔹 RMSE: {rmse:.4f}")
        print(f"🔹 MAE: {mae:.4f}")
        
        # Save artifacts
        joblib.dump(scaler, f"app/ml/{model_type}_scaler.pkl")
        joblib.dump(model, f"app/ml/{model_type}_model.pkl")
        
    print("\n✅ All models trained and saved successfully!")

if __name__ == "__main__":
    train_models()