import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
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
driver_standings = data["driver_standings"].copy()
constructor_standings = data["constructor_standings"].copy()  # Fixed reference

# ✅ Merge races first, with circuitId
merged_df = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

# ✅ Merge all relevant data
merged_df = merged_df.merge(drivers, on="driverId", how="left")
merged_df = merged_df.merge(circuits, on="circuitId", how="left")
merged_df = merged_df.merge(constructors, on="constructorId", how="left")

# ✅ Compute average pit stop time per race
pit_stop_avg = pit_stops.groupby("raceId")["milliseconds"].mean().reset_index()
pit_stop_avg.rename(columns={"milliseconds": "avg_pit_time"}, inplace=True)
merged_df = merged_df.merge(pit_stop_avg, on="raceId", how="left")

# ✅ Compute track-specific average lap time
lap_times = lap_times.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

# ✅ Compute average qualifying time
qualifying = qualifying.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

for col in ["q1", "q2", "q3"]:
    qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")

qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
qualifying_avg = qualifying.groupby(["driverId", "circuitId"])["avg_qualifying_time"].mean().reset_index()
merged_df = merged_df.merge(qualifying_avg, on=["driverId", "circuitId"], how="left")

# ✅ Merge standings
driver_standings = driver_standings[["raceId", "driverId", "points", "position"]].copy()
driver_standings.rename(columns={"points": "driver_points", "position": "driver_position"}, inplace=True)
merged_df = merged_df.merge(driver_standings, on=["raceId", "driverId"], how="left")

constructor_standings = constructor_standings[["raceId", "constructorId", "points", "position"]].copy()
constructor_standings.rename(columns={"points": "constructor_points", "position": "constructor_position"}, inplace=True)
merged_df = merged_df.merge(constructor_standings, on=["raceId", "constructorId"], how="left")

# ✅ Check for NaN or infinite values
print("\n🛠️ Checking for NaN or infinite values:")
print(merged_df.isna().sum())
print(merged_df.isin([np.inf, -np.inf]).sum())

# ✅ Fill missing values and replace infinite values
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())
merged_df[numeric_cols] = merged_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(merged_df[numeric_cols].median())

# ✅ Select features for training
features = [
    "grid", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
    "driver_points", "driver_position", "constructor_points", "constructor_position"
]

# Ensure target column exists and normalize by max position count
if "positionOrder" in merged_df.columns:
    y = merged_df["positionOrder"] / merged_df["positionOrder"].max()  # Dynamic normalization
else:
    raise ValueError("Column 'positionOrder' not found in dataset.")

X = merged_df[features]

# ✅ Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "app/ml/scaler.pkl")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Initialize the XGBRegressor model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ✅ Fit the model without early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",  # Use RMSE for regression
    verbose=True
)

# ✅ Save the trained model
joblib.dump(model, "app/ml/f1_xgb_model.pkl")
print("\n✅ Model training complete!")

# ✅ Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n📊 **Model Performance:**")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
