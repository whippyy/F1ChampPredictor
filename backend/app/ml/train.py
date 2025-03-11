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
constructor_standings = data["standings"].copy()

# ✅ Merge races first, with circuitId
merged_df = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left")

# ✅ Merge all relevant data
merged_df = merged_df.merge(drivers, on="driverId", how="left")
merged_df = merged_df.merge(circuits, on="circuitId", how="left")
merged_df = merged_df.merge(constructors, on="constructorId", how="left")

# ✅ Compute average lap and pit times
lap_times = lap_times.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
avg_lap_time = lap_times.groupby(["driverId", "circuitId"])["milliseconds"].mean().reset_index()
avg_lap_time.rename(columns={"milliseconds": "avg_lap_time"}, inplace=True)
merged_df = merged_df.merge(avg_lap_time, on=["driverId", "circuitId"], how="left")

pit_stop_avg = pit_stops.groupby("raceId")["milliseconds"].mean().reset_index().rename(columns={"milliseconds": "avg_pit_time"})
merged_df = merged_df.merge(pit_stop_avg, on="raceId", how="left")

# ✅ Compute average qualifying time
qualifying = qualifying.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
for col in ["q1", "q2", "q3"]:
    qualifying[col] = pd.to_numeric(qualifying[col], errors="coerce")
qualifying["avg_qualifying_time"] = qualifying[["q1", "q2", "q3"]].mean(axis=1)
qualifying_avg = qualifying.groupby(["driverId", "circuitId"])["avg_qualifying_time"].mean().reset_index()
merged_df = merged_df.merge(qualifying_avg, on=["driverId", "circuitId"], how="left")

# ✅ Merge standings
driver_standings.rename(columns={"points": "driver_points", "position": "driver_position"}, inplace=True)
merged_df = merged_df.merge(driver_standings[["raceId", "driverId", "driver_points", "driver_position"]], on=["raceId", "driverId"], how="left")

constructor_standings.rename(columns={"points": "constructor_points", "position": "constructor_position"}, inplace=True)
merged_df = merged_df.merge(constructor_standings[["raceId", "constructorId", "constructor_points", "constructor_position"]], on=["raceId", "constructorId"], how="left")

# ✅ Fill missing values using circuit medians
numeric_cols = ["avg_lap_time", "avg_pit_time", "avg_qualifying_time", "driver_points", "driver_position", "constructor_points", "constructor_position"]
for col in numeric_cols:
    merged_df[col] = merged_df.groupby("circuitId")[col].transform(lambda x: x.fillna(x.median()))
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())  # Fill remaining NaNs

# ✅ Select features for training
features = [
    "grid", "avg_lap_time", "avg_pit_time", "avg_qualifying_time",
    "driver_points", "driver_position", "constructor_points", "constructor_position"
]

y = merged_df["positionOrder"] / 20.0  # Normalize target
X = merged_df[features]

# ✅ Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "app/ml/scaler.pkl")

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Initialize XGBRegressor model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  
    early_stopping_rounds=10,  
    eval_metric="rmse",  
    verbose=True
)


# ✅ Fit the model with early stopping only if `y_test` is valid
if len(set(y_test)) > 1:
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,  # Stops training if no improvement after 10 rounds
        eval_metric="rmse",
        verbose=True
    )
else:
    print("⚠️ Not enough variation in y_test for early stopping. Training without it.")
    model.fit(X_train, y_train)

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
