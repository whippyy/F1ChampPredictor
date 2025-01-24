from fastapi import FastAPI
from app.routes import data, drivers, predictions, teams

app = FastAPI()

app.include_router(data.router)
app.include_router(drivers.router)
app.include_router(predictions.router)
app.include_router(teams.router)




