import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
constructors = data["constructors"]
driver_standings = data["driver_standings"]
constructor_standings = data["standings"]

# ✅ Merge races first, with circuitId
merged_df = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left", suffixes=("_result", "_race"))

# ✅ Merge all the relevant data into one dataframe
merged_df = merged_df.merge(drivers, on="driverId", how="left", suffixes=("_race", "_driver"))
merged_df = merged_df.merge(circuits, on="circuitId", how="left", suffixes=("_driver", "_circuit"))
merged_df = merged_df.merge(constructors, on="constructorId", how="left", suffixes=("_circuit", "_constructor"))
merged_df = merged_df.merge(pit_stops.groupby("raceId").agg({'milliseconds':'mean'}).rename(columns={'milliseconds':'avg_pit_time'}), on="raceId", how="left", suffixes=("_constructor", "_pit"))

# ✅ Ensure circuitId exists in lap_times before grouping
if "circuitId" not in lap_times.columns:
    lap_times = lap_times.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

# ✅ Compute track-specific average lap time
avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

# ✅ Ensure circuitId exists in qualifying before grouping
if "circuitId" not in qualifying.columns:
    qualifying = qualifying.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

# ✅ Convert qualifying times to numeric values (handling missing values)
for col in ["q1", "q2", "q3"]:
    qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")

# ✅ Compute average qualifying time per driver per track
qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)

# ✅ Merge into merged_df
qualifying_avg = qualifying.groupby(["driverId", "circuitId"])["avg_qualifying_time"].mean().reset_index()
merged_df = merged_df.merge(qualifying_avg, on=["driverId", "circuitId"], how="left")

# ✅ Compute track-specific qualifying position
driver_circuit_grid = qualifying.groupby(["driverId", "circuitId"])["position"].mean().reset_index()
driver_circuit_grid.rename(columns={"position": "grid_position"}, inplace=True)
merged_df = merged_df.merge(driver_circuit_grid, on=["driverId", "circuitId"], how="left")

# ✅ Merge additional driver and constructor standings info
merged_df = merged_df.merge(driver_standings, on=["raceId", "driverId"], how="left", suffixes=("", "_driver_standing"))
merged_df = merged_df.merge(constructor_standings, on=["raceId", "constructorId"], how="left", suffixes=("", "_constructor_standing"))

# ✅ Ensure only numeric columns are used for median calculations
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# ✅ Select features for training - Using more features from the merged data
features = [
    "grid_position", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
    "positionOrder", "driver_standing", "constructor_standing"
]

X = merged_df[features]
y = merged_df["positionOrder"] / 20.0  # Normalize target (assuming 20 as max positions)

# ✅ Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "app/ml/scaler.pkl")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Define and train the model with dropout
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),  # Dropout layer to reduce overfitting
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')  # Linear activation for regression
])

# ✅ Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# ✅ Train the model with callbacks
history = model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# ✅ Save the trained model
model.save("app/ml/f1_model.keras")
print("✅ Model training complete!")

# ✅ Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# ✅ Model summary
model.summary()
