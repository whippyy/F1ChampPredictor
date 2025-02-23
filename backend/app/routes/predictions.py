from fastapi import APIRouter
from app.ml.predict import predict_race
from app.schemas import PredictionRequest  # ✅ Ensure correct schema name

router = APIRouter()

@router.post("/predict")
def predict_race_api(data: PredictionRequest):
    prediction_result = predict_race(
        driver_id=data.driver_id,
        circuit_id=data.circuit_id,
        grid=data.grid,
        points=data.points,
        fastest_lap=data.fastest_lap,
        qualifying_position=data.qualifying_position,  # ✅ Add missing fields
        avg_qualifying_time=data.avg_qualifying_time   # ✅ Add missing fields
    )
    return prediction_result







