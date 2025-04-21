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
    """Prepare features focusing on driver-circuit performance history"""
    print("🟡 Preparing features...")
    
    # Load all data files
    drivers = data["drivers"]
    results = data["results"]
    races = data["races"]
    circuits = data["circuits"]
    lap_times = data["lap_times"]
    pit_stops = data["pit_stops"]
    qualifying = data["qualifying"]
    driver_standings = data["driver_standings"]
    constructor_standings = data["standings"]
    constructors = data["constructors"]

    # Create base dataframe with race and circuit info
    races_with_circuits = races[['raceId', 'circuitId', 'year', 'round']]
    
    # 1. Driver's historical performance at each circuit
    driver_circuit_history = (
        results
        .merge(races_with_circuits, on='raceId')
        .groupby(['driverId', 'circuitId'])
        .agg(
            driver_circuit_races=('raceId', 'count'),
            driver_circuit_avg_finish=('positionOrder', 'mean'),
            driver_circuit_best_finish=('positionOrder', 'min'),
            driver_circuit_top3_rate=('positionOrder', lambda x: (x <= 3).mean()),
            driver_circuit_avg_points=('points', 'mean')
        )
        .reset_index()
    )
    
    # 2. Current race performance metrics
    merged_df = (
        results
        .merge(races_with_circuits, on='raceId')
        .merge(drivers, on='driverId')
        .merge(constructors, on='constructorId')
    )
    
    # 3. Qualifying performance (relative to others in same race)
    qualifying_perf = (
        qualifying
        .merge(races_with_circuits, on='raceId')
        .groupby('raceId')
        .apply(lambda x: x.assign(
            quali_percentile=x['position'].rank(pct=True)
        ))
        [['raceId', 'driverId', 'quali_percentile']]
    )
    
    # 4. Recent form (last 5 races performance)
    recent_form = (
        results
        .merge(races[['raceId', 'date']], on='raceId')
        .sort_values(['driverId', 'date'])
        .groupby('driverId')
        .tail(5)
        .groupby('driverId')
        .agg(
            recent_avg_finish=('positionOrder', 'mean'),
            recent_avg_points=('points', 'mean')
        )
        .reset_index()
    )
    
    # Combine all features
    final_df = (
        merged_df
        .merge(driver_circuit_history, on=['driverId', 'circuitId'], how='left')
        .merge(qualifying_perf, on=['raceId', 'driverId'], how='left')
        .merge(recent_form, on='driverId', how='left')
        .merge(driver_standings, on=['raceId', 'driverId'], how='left')
        .merge(constructor_standings, on=['raceId', 'constructorId'], how='left')
    )
    
    # Feature engineering
    final_df['grid_to_quali_ratio'] = final_df['grid'] / (final_df['position'] + 1)
    final_df['points_per_race'] = final_df['points'] / final_df['driver_circuit_races']
    
    # Handle missing values
    fill_values = {
        'driver_circuit_races': 0,
        'driver_circuit_avg_finish': 15,
        'driver_circuit_best_finish': 20,
        'driver_circuit_top3_rate': 0,
        'driver_circuit_avg_points': 0,
        'quali_percentile': 0.5,
        'recent_avg_finish': 15,
        'recent_avg_points': 0,
        'grid_to_quali_ratio': 1,
        'points_per_race': 0
    }
    final_df = final_df.fillna(fill_values)
    
    return final_df

def train_models():
    print("🟡 Starting model training...")
    data = load_csv_data()
    df = prepare_features(data)
    
    # Feature selection - focus on driver/circuit history and performance
    race_features = [
        'grid',
        'quali_percentile',
        'driver_circuit_races',
        'driver_circuit_avg_finish',
        'driver_circuit_best_finish',
        'driver_circuit_top3_rate',
        'driver_circuit_avg_points',
        'recent_avg_finish',
        'recent_avg_points',
        'grid_to_quali_ratio',
        'points_per_race',
        'points',  # Current race points
        'position'  # Current race position
    ]
    
    X = df[race_features]
    y = df['positionOrder']  # Actual finishing position
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
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
    
    print(f"\n📊 Model Performance:")
    print(f"🔹 RMSE: {rmse:.4f}")
    print(f"🔹 MAE: {mae:.4f}")
    
    # Save model
    joblib.dump(model, "app/ml/f1_driver_model.pkl")
    joblib.dump(scaler, "app/ml/f1_scaler.pkl")
    
    print("\n✅ Model trained and saved successfully!")

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