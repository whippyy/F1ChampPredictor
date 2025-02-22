import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("app/ml/f1_model.h5")

def predict_race(grid: int, points: float, dob: int):
    input_data = np.array([[grid, points, dob]])
    prediction = model.predict(input_data)
    return int(prediction[0][0])


