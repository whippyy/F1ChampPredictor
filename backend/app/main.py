from fastapi import FastAPI
from app.routes import data, drivers, predictions, teams

app = FastAPI()

# Add a simple root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the F1ChampPredictor API!"}

# Include other routers
app.include_router(data.router)
app.include_router(drivers.router)
app.include_router(predictions.router)
app.include_router(teams.router)






