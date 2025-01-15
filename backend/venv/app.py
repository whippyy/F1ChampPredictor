from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    driver: str
    race: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the F1 Prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Placeholder for the prediction logic
    return {"driver": request.driver, "race": request.race, "prediction": "Win"}
