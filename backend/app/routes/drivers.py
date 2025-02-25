from fastapi import APIRouter, Query
from app.data_loader import load_csv_data

router = APIRouter()

@router.get("/drivers")
def get_current_drivers(team_id: int = Query(None, description="Filter drivers by team")):
    """Fetch the 20 drivers for the current season, with optional team filtering."""
    data = load_csv_data()
    drivers_df = data["drivers"]
    results_df = data["results"]
    races_df = data["races"]

    # Get the most recent season
    current_season = races_df["year"].max()

    # Get all drivers who participated in the most recent season
    current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
    current_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]

    if team_id is not None:
        current_drivers = current_drivers[current_drivers["constructorId"] == team_id]

    unique_driver_ids = current_drivers["driverId"].unique()
    filtered_drivers = drivers_df[drivers_df["driverId"].isin(unique_driver_ids)]

    return {
        "message": "Current season drivers fetched successfully",
        "data": filtered_drivers.to_dict(orient="records")
    }










