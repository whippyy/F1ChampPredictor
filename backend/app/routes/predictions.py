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

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Prepare feature vector
    feature_vector = prepare_feature_vector(request.driver_id, request.team_id)
    
    if feature_vector:
        # Make prediction
        prediction = model.predict([feature_vector])
        return PredictionResponse(prediction=str(prediction[0]))
    else:
        return {"error": "Could not fetch stats or prepare feature vector."}


