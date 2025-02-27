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
standings = data["driver_standings"]
pit_stops = data["pit_stops"]
qualifying = data["qualifying"]

# ✅ Merge relevant tables
merged_df = results.merge(races, on="raceId", how="left")
merged_df = merged_df.merge(drivers, on="driverId", how="left")
merged_df = merged_df.merge(circuits, on="circuitId", how="left")
merged_df = merged_df.merge(standings, on=["driverId", "raceId"], how="left")
merged_df = merged_df.merge(pit_stops.groupby("raceId").agg({'milliseconds':'mean'}).rename(columns={'milliseconds':'avg_pit_time'}), on="raceId", how="left")

print("🔍 Available columns in merged_df:", merged_df.columns.tolist())

# Ensure `race_points` comes from results
merged_df["race_points"] = merged_df["race_points"]  # Use renamed column
merged_df["season_points"] = merged_df["season_points"]

# Ensure `season_points` comes from standings
merged_df["season_points"] = merged_df["points_y"]  # Use season-long accumulated points

# Normalize points by max season points to reduce dominance
max_season_points = merged_df["season_points"].max()
merged_df["race_points"] = merged_df["race_points"] / max_season_points
merged_df["season_points"] = merged_df["season_points"] / max_season_points

# ✅ Compute average lap time per driver per race
avg_lap_time = lap_times.groupby(["raceId", "driverId"])['milliseconds'].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
merged_df = merged_df.merge(avg_lap_time, on=["raceId", "driverId"], how="left")

# ✅ Process qualifying times
for col in ["q1", "q2", "q3"]:
    qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")
qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
merged_df = merged_df.merge(qualifying[["raceId", "driverId", "avg_qualifying_time"]], on=["raceId", "driverId"], how="left")

# Ensure only numeric columns are used for median calculations
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# Ensure correct column names before renaming
if "points_x" in merged_df.columns and "points_y" in merged_df.columns:
    merged_df.rename(columns={"points_x": "race_points", "points_y": "season_points"}, inplace=True)
elif "points" in merged_df.columns:
    merged_df["race_points"] = merged_df["points"]
    merged_df["season_points"] = merged_df["points"]
else:
    raise KeyError("⚠️ 'points' column not found in merged_df! Check merged data.")


# ✅ Select features for training
features = ["grid", "race_points", "season_points", "fastestLapSpeed", "avg_lap_time", "avg_pit_time", "avg_qualifying_time"]
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

