import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "app/ml/f1_model.keras"

# Check if the model exists before loading
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    model = None
    print("⚠️ Model file not found! Train the model first.")

def predict_race(grid: int, points: float, dob: int):
    if model is None:
        raise ValueError("No trained model found! Please train the model first.")

    input_data = np.array([[grid, points, dob]])
    prediction = model.predict(input_data)
    return int(prediction[0][0])


