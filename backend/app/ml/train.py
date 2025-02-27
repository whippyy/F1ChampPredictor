import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from app.data_loader import load_csv_data
import joblib

# ✅ Load all data files
data = load_csv_data()
drivers = data["drivers"]
results = data["results"]
races = data["races"]
circuits = data["circuits"]
lap_times = data["lap_times"]
pit_stops = data["pit_stops"]
qualifying = data["qualifying"]

# ✅ Merge relevant tables
merged_df = results.merge(races, on="raceId", how="left")
merged_df = merged_df.merge(drivers, on="driverId", how="left")
merged_df = merged_df.merge(circuits, on="circuitId", how="left")
merged_df = merged_df.merge(pit_stops.groupby("raceId").agg({'milliseconds':'mean'}).rename(columns={'milliseconds':'avg_pit_time'}), on="raceId", how="left")

# ✅ Compute track-specific average lap time
# ✅ First, merge lap_times with races to get circuitId
lap_times = lap_times.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

# ✅ Now, compute the track-specific average lap time
avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

# ✅ Process track-specific qualifying positions
driver_circuit_grid = qualifying.groupby(["driverId", "circuitId"])["position"].mean().reset_index()
driver_circuit_grid.rename(columns={"position": "grid_position"}, inplace=True)
merged_df = merged_df.merge(driver_circuit_grid, on=["driverId", "circuitId"], how="left")

# Ensure only numeric columns are used for median calculations
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# ✅ Select features for training
features = ["grid_position", "avg_lap_time", "avg_pit_time", "avg_qualifying_time"]
X = merged_df[features]
y = merged_df["positionOrder"] / 20.0  # Normalize target

# ✅ Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "app/ml/scaler.pkl")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Define and train the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))

# ✅ Save the trained model
model.save("app/ml/f1_model.keras")
print("✅ Model training complete!")
