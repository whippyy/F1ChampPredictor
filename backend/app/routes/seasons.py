from fastapi import APIRouter
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/seasons")
def get_seasons():
    """Fetch available seasons from races dataset."""
    data = load_csv_data()
    races_df = data["races"]
    
    # Get unique seasons
    seasons = races_df["year"].unique().tolist()
    
    return {"message": "Seasons fetched successfully", "data": seasons}