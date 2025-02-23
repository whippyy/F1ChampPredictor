import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.data_loader import load_csv_data

# Load data
data = load_csv_data()
drivers = data["drivers"]
results = data["results"]
circuits = data["circuits"]
lap_times = data["lap_times"]

# Merge results with drivers & circuits
df = results.merge(drivers, on="driverId").merge(circuits, on="circuitId", how="left")

# Get fastest lap per race
fastest_laps = lap_times.groupby(["raceId", "driverId"])["milliseconds"].min().reset_index()
df = df.merge(fastest_laps, on=["raceId", "driverId"], how="left")

# Select relevant features
features = df[["grid", "points", "milliseconds", "dob", "fastestLapSpeed"]]
target = df["positionOrder"]

# Encode categorical data (e.g., dob)
encoder = LabelEncoder()
features["dob"] = encoder.fit_transform(features["dob"])

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Save the scaler for predictions
import joblib
joblib.dump(scaler, "app/ml/scaler.pkl")

# Define a better model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train / 20.0, epochs=100, batch_size=16, validation_data=(X_test, y_test / 20.0))

# Save the trained model
model.save("app/ml/f1_model.keras")
print("✅ Model training complete!")