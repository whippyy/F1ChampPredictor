import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.data_loader import load_csv_data
import joblib
from datetime import datetime

# ✅ Load data
data = load_csv_data()
drivers = data["drivers"]
results = data["results"]
races = data["races"]  # ✅ Load races to get `circuitId`
circuits = data["circuits"]
lap_times = data["lap_times"]
qualifying = data["qualifying"]

# ✅ Merge results with races to get `circuitId`
df = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left") \
            .merge(drivers, on="driverId", how="left") \
            .merge(circuits[["circuitId", "alt"]], on="circuitId", how="left")  # Track altitude

# ✅ Convert qualifying lap times to milliseconds
def time_to_milliseconds(time_str):
    """Convert '1:26.572' to total milliseconds"""
    if pd.isna(time_str) or time_str == "\\N":
        return np.nan  # Handle missing values
    minutes, seconds = time_str.split(":")
    return int(minutes) * 60000 + float(seconds) * 1000

# ✅ Apply conversion for q1, q2, q3
for col in ["q1", "q2", "q3"]:
    qualifying[col] = qualifying[col].apply(time_to_milliseconds)

# ✅ Compute average qualifying time
qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)

# ✅ Merge qualifying position & lap times into dataset
df = df.merge(
    qualifying[["raceId", "driverId", "position", "avg_qualifying_time"]],
    on=["raceId", "driverId"], 
    how="left"
)
# Convert 'dob' to age
current_year = datetime.now().year
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = current_year - df["dob"].dt.year  # Convert to age

# Drop original 'dob' column
df.drop(columns=["dob"], inplace=True)

# ✅ Rename column for clarity
df.rename(columns={"position_y": "qualifying_position"}, inplace=True)

# ✅ Compute average lap time per race
avg_lap_time = lap_times.groupby(["raceId", "driverId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
df = df.merge(avg_lap_time, on=["raceId", "driverId"], how="left")

# ✅ Handle missing values
df.replace("\\N", np.nan, inplace=True)  # Convert '\N' to NaN
df.dropna(inplace=True)  # Drop rows with missing values

# ✅ Select relevant features
features = df[["grid", "points", "age", "fastestLapSpeed", "qualifying_position", "avg_lap_time", "alt", "avg_qualifying_time"]].copy()

# ✅ Convert to numeric & handle missing values
features["qualifying_position"] = pd.to_numeric(features["qualifying_position"], errors="coerce")
features["avg_qualifying_time"] = pd.to_numeric(features["avg_qualifying_time"], errors="coerce")
features.fillna(features.mean(), inplace=True)

# ✅ Encode categorical data (Convert `dob` to numerical values)
encoder = LabelEncoder()
features["dob"] = encoder.fit_transform(features["dob"])

# ✅ Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ✅ Save the scaler for predictions
joblib.dump(scaler, "app/ml/scaler.pkl")

# ✅ Prepare target variable (final race position)
target = df["positionOrder"]

# ✅ Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# ✅ Define improved model with deeper architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output between 0 and 1
])

# ✅ Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ Train model with more epochs
model.fit(X_train, y_train / 20.0, epochs=150, batch_size=32, validation_data=(X_test, y_test / 20.0))

# ✅ Save the trained model
model.save("app/ml/f1_model.keras")
print("✅ Model training complete!")

