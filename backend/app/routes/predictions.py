from fastapi import APIRouter
from app.ml.predict import predict_race
from app.schemas import PredictionInput

router = APIRouter()

@router.post("/predict")
def predict_race_api(data: PredictionInput):
    prediction_result = predict_race(
        driver_id=data.driver_id,
        track_id=data.track_id,
        grid=data.grid,
        points=data.points,
        dob=data.dob
    )
    return prediction_result






