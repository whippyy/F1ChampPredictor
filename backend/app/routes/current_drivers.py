from fastapi import APIRouter
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/current_drivers")
def get_current_drivers():
    """Fetches the drivers for the current season"""
    data = load_csv_data()
    drivers_df = data["drivers"]

    return {
        "message": "Current season drivers fetched successfully",
        "data": drivers_df.to_dict(orient="records")
    }