from fastapi import FastAPI
from app.routes import data, drivers, predictions, teams

app = FastAPI()

# Include routers
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(drivers.router, prefix="/drivers", tags=["Drivers"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(teams.router, prefix="/teams", tags=["Teams"])



