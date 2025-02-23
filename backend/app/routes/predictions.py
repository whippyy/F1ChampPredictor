from fastapi import APIRouter
from app.ml.predict import predict_race
from app.schemas import PredictionInput

router = APIRouter()

@router.post("/predict")
def predict_race_api(data: PredictionInput):
    prediction_result = predict_race(
        driver_id=data.driver_id,
        circuit_id=data.circuit_id,
        grid=data.grid,
        points=data.points,
        dob=data.dob,
        fastest_lap=data.fastest_lap
    )
    return prediction_result






