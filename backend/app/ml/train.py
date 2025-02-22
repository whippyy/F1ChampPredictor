import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from app.data_loader import load_csv_data

# Load data
data = load_csv_data()
df = data["results"].merge(data["drivers"], on="driverId").merge(data["races"], on="raceId")

# Select features & target
features = df[["grid", "points", "dob"]]
target = df["positionOrder"]

# Encode categorical data
encoder = LabelEncoder()
features["dob"] = encoder.fit_transform(features["dob"])

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)  # Predict position
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save("app/ml/f1_model.keras")
print("Model training complete and saved as f1_model.h5")
