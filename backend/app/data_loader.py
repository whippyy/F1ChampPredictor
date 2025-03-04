import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Load all data files using your data loader
def load_csv_data():
    return {
        'circuits': pd.read_csv('app/data/circuits.csv', na_values=["\\N"]),
        'constructors_results': pd.read_csv('app/data/constructor_results.csv', na_values=["\\N"]),
        'standings': pd.read_csv('app/data/constructor_standings.csv', na_values=["\\N"]),
        'constructors': pd.read_csv('app/data/constructors.csv', na_values=["\\N"]),
        'driver_standings': pd.read_csv('app/data/driver_standings.csv', na_values=["\\N"]),
        'drivers': pd.read_csv('app/data/drivers.csv', na_values=["\\N"]),
        'lap_times': pd.read_csv('app/data/lap_times.csv', na_values=["\\N"]),
        'pit_stops': pd.read_csv('app/data/pit_stops.csv', na_values=["\\N"]),
        'qualifying': pd.read_csv('app/data/qualifying.csv', na_values=["\\N"]),
        'races': pd.read_csv('app/data/races.csv', na_values=["\\N"]),
        'results': pd.read_csv('app/data/results.csv', na_values=["\\N"]),
        'seasons': pd.read_csv('app/data/seasons.csv', na_values=["\\N"]),
        'sprint_results': pd.read_csv('app/data/sprint_results.csv', na_values=["\\N"]),
        'status': pd.read_csv('app/data/status.csv', na_values=["\\N"])
    }

def get_data(file_name: str):
    data = load_csv_data()
    return data.get(file_name)

# ✅ Merge all the relevant data tables
def prepare_data():
    # Load the data
    drivers = get_data("drivers")
    results = get_data("results")
    races = get_data("races")
    circuits = get_data("circuits")
    lap_times = get_data("lap_times")
    pit_stops = get_data("pit_stops")
    qualifying = get_data("qualifying")

    # Merge relevant tables
    merged_df = results.merge(races, on="raceId", how="left")
    merged_df = merged_df.merge(drivers, on="driverId", how="left")
    merged_df = merged_df.merge(circuits, on="circuitId", how="left")
    merged_df = merged_df.merge(pit_stops.groupby("raceId").agg({'milliseconds':'mean'}).rename(columns={'milliseconds':'avg_pit_time'}), on="raceId", how="left")

    # Compute track-specific average lap time
    avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
    avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
    merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

    # Compute average qualifying time per driver per track
    qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
    qualifying_avg = qualifying.groupby(["driverId", "circuitId"])["avg_qualifying_time"].mean().reset_index()
    merged_df = merged_df.merge(qualifying_avg, on=["driverId", "circuitId"], how="left")

    # Compute track-specific qualifying position
    driver_circuit_grid = qualifying.groupby(["driverId", "circuitId"])["position"].mean().reset_index()
    driver_circuit_grid.rename(columns={"position": "grid_position"}, inplace=True)
    merged_df = merged_df.merge(driver_circuit_grid, on=["driverId", "circuitId"], how="left")

    # Ensure numeric columns are filled with median values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

    return merged_df

# ✅ Define features for training
def prepare_features_and_target(df):
    features = ["grid_position", "avg_lap_time", "avg_pit_time", "avg_qualifying_time"]
    X = df[features]
    y = df["positionOrder"] / 20.0  # Normalize target (if positionOrder is the race finishing position)

    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "app/ml/scaler.pkl")

    return X_scaled, y

# ✅ Define and train the model
def train_model(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=150, batch_size=32)
    model.save("app/ml/f1_model.keras")
    print("✅ Model training complete!")

# ✅ Main function
def main():
    # Step 1: Prepare data
    merged_df = prepare_data()

    # Step 2: Prepare features and target
    X, y = prepare_features_and_target(merged_df)

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the model
    train_model(X_train, y_train)

if __name__ == "__main__":
    main()

