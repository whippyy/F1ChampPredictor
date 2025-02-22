from .data import router as data_router
from .drivers import router as drivers_router
from .teams import router as teams_router
from .predictions import router as predictions_router

# Expose routers for easy import in main.py
__all__ = ["data_router", "drivers_router", "teams_router", "predictions_router"]

