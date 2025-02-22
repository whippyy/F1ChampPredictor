from fastapi import APIRouter
from pydantic import BaseModel
from app.ml.predict import predict_race

router = APIRouter()

class PredictionInput(BaseModel):
    grid: int
    points: float
    dob: int

@router.post("/predict")
def predict_race_api(data: PredictionInput):
    predicted_position = predict_race(data.grid, data.points, data.dob)
    return {"predicted_position": predicted_position}





