from fastapi import FastAPI
from app.routes import data, predictions

app = FastAPI()

# Register routers
app.include_router(data.router)
app.include_router(predictions.router)


