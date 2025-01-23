# app/routes/predictions.py
from fastapi import APIRouter
from pydantic import BaseModel
import joblib
from typing import List
from app.predict import prepare_feature_vector

# Load the trained model
model = joblib.load('f1_model.pkl')

# Define request model for prediction input
class PredictionRequest(BaseModel):
    driver_id: int  # e.g., ID of the driver
    team_id: int    # e.g., ID of the team
    race_id: int    # e.g., ID of the race
    historical_data: List[float]  # any historical stats or features needed for prediction

class PredictionResponse(BaseModel):
    prediction: str

router = APIRouter()

@router.get("/predict")
def predict(driver_id: int, team_id: int):
    feature_vector = prepare_feature_vector(driver_id, team_id)
    if feature_vector:
        # Run the model prediction here with feature_vector
        prediction = "some prediction logic"  # Replace with actual model prediction
        return {"prediction": prediction}
    return {"error": "Failed to prepare feature vector"}


