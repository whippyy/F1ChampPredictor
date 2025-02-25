from fastapi import APIRouter, Query
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/races")
def get_races(season: int = Query(..., description="F1 season year")):
    """Fetch all races for a given season."""
    data = load_csv_data()
    races_df = data["races"]

    # Filter races for the given season
    season_races = races_df[races_df["year"] == season]

    if season_races.empty:
        return {"message": f"No races found for season {season}", "data": []}

    return {
        "message": f"Races for season {season} fetched successfully",
        "data": season_races.to_dict(orient="records")
    }
