from fastapi import APIRouter
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/current_circuits")
def get_current_circuits():
    """Fetches the circuits for the current season"""
    data = load_csv_data()
    circuits_df = data["circuits"]

    return {
        "message": "Current season circuits fetched successfully",
        "data": circuits_df.to_dict(orient="records")
    }
