from fastapi import FastAPI
from app.routes import predictions, drivers, teams

app = FastAPI(
    title="F1 Championship Predictor",
    description="An API to predict F1 championship outcomes using machine learning.",
    version="1.0.0",
)

# Include the route modules
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(drivers.router, prefix="/drivers", tags=["Drivers"])
app.include_router(teams.router, prefix="/teams", tags=["Teams"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the F1 Championship Predictor API!"}

