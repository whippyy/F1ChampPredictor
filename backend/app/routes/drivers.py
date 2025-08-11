from fastapi import APIRouter, Query, HTTPException
from app.data_loader import f1_data
from app.schemas import Driver
from typing import List

router = APIRouter()

@router.get("/drivers", response_model=List[Driver], tags=["Data Fetching"])
def get_current_drivers(team_id: int = Query(None, description="Filter drivers by team")) -> List[Driver]:
    """Fetch the 20 drivers for the current season, with optional team filtering."""
    try:
        drivers_df = f1_data.data["drivers"]
        results_df = f1_data.data["results"]

        if team_id is not None:
            # Filter results for the specified team in the current season
            current_season_results = results_df[results_df["raceId"].isin(f1_data.current_race_ids)]
            team_drivers = current_season_results[current_season_results["constructorId"] == team_id]
            driver_ids = team_drivers["driverId"].unique()
        else:
            # Get all drivers for the current season
            driver_ids = f1_data.current_driver_ids

        filtered_drivers = drivers_df[drivers_df["driverId"].isin(driver_ids)]

        return filtered_drivers.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching drivers: {str(e)}")
