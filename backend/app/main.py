from fastapi import FastAPI
from app.routes import data, drivers, predictions, teams, seasons, races
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Enable CORS for frontend (React at localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Add a simple root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the F1ChampPredictor API!"}



# Include other routers
app.include_router(data.router)
app.include_router(drivers.router)
app.include_router(predictions.router)
app.include_router(teams.router)
app.include_router(seasons.router)
app.include_router(races.router)






