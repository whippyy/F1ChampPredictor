from fastapi import APIRouter
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/teams")
def get_current_teams():
    """Fetch all teams (constructors) that participated in the current season."""
    data = load_csv_data()
    constructors_df = data["constructors"]
    results_df = data["results"]
    races_df = data["races"]

    # Get the most recent season
    current_season = races_df["year"].max()

    # Get all teams that participated in the most recent season
    current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
    current_teams = results_df[results_df["raceId"].isin(current_season_race_ids)]["constructorId"].unique()

    filtered_teams = constructors_df[constructors_df["constructorId"].isin(current_teams)]

    return {
        "message": "Current season teams fetched successfully",
        "data": filtered_teams.to_dict(orient="records")
    }





