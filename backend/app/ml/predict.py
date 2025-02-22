import joblib

# Load a pre-trained model
model = joblib.load("path_to_your_model.pkl")

def predict(driver_stats):
    return model.predict([driver_stats])


