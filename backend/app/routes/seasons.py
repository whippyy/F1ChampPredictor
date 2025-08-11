from fastapi import APIRouter, HTTPException
from app.data_loader import f1_data
from typing import List

router = APIRouter()

@router.get("/seasons", response_model=List[int], tags=["Data Fetching"])
def get_seasons() -> List[int]:
    """Fetch available seasons from races dataset."""
    try:
        races_df = f1_data.data["races"]
        
        # Get unique seasons and sort them descending
        seasons = sorted(races_df["year"].unique().tolist(), reverse=True)
        return seasons
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching seasons: {str(e)}")