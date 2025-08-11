from fastapi import APIRouter, Query, HTTPException
from app.data_loader import f1_data
from app.schemas import Race
from typing import List

router = APIRouter()

@router.get("/races", response_model=List[Race], tags=["Data Fetching"])
def get_races(season: int = Query(..., description="F1 season year")) -> List[Race]:
    """Fetch all races for a given season."""
    try:
        races_df = f1_data.data["races"]

        # Filter races for the given season
        season_races = races_df[races_df["year"] == season]

        if season_races.empty:
            return []

        # Replace NaN values with None for Pydantic compatibility
        season_races = season_races.where
