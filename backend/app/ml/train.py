import pandas as pd
import numpy as np
import tensorflow as tf
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

# ✅ Define placeholders for TensorFlow 1.x
tf.reset_default_graph()  # Clear any existing graph
X_placeholder = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
y_placeholder = tf.placeholder(tf.float32, shape=(None, 1))

# ✅ Define the model
hidden1 = tf.layers.dense(X_placeholder, 128, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(hidden1, rate=0.2)
hidden2 = tf.layers.dense(dropout1, 64, activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, 32, activation=tf.nn.relu)
output = tf.layers.dense(hidden3, 1, activation=None)

# ✅ Define loss and optimizer
loss = tf.losses.mean_squared_error(y_placeholder, output)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# ✅ Initialize variables
init = tf.global_variables_initializer()

# ✅ Train the model
with tf.Session() as sess:
    sess.run(init)
    
    # Training loop
    for epoch in range(150):
        _, train_loss = sess.run([optimizer, loss], feed_dict={X_placeholder: X_train, y_placeholder: y_train.values.reshape(-1, 1)})
        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss}")
    
    # Evaluate the model
    y_pred = sess.run(output, feed_dict={X_placeholder: X_test})
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)

    # Save the model
    saver = tf.train.Saver()
    saver.save(sess, "app/ml/f1_model.ckpt")
    print("✅ Model training complete!")
