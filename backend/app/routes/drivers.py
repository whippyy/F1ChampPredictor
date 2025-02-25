from fastapi import APIRouter
from app.data_loader import load_csv_data
from pydantic import BaseModel

# Define response model
class DriverResponse(BaseModel):
    message: str
    data: list

router = APIRouter()

@router.get("/drivers", tags=["Data Fetching"], response_model=DriverResponse)
def get_current_drivers():
    """Fetch the 20 drivers for the current season."""
    try:
        data = load_csv_data()
        drivers_df = data["drivers"]
        results_df = data["results"]
        races_df = data["races"]

        # ✅ Get the most recent season
        current_season = races_df["year"].max()

        # ✅ Get all drivers who participated in the most recent season
        current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
        current_drivers = results_df[results_df["raceId"].isin(current_season_race_ids)]["driverId"].unique()
        
        # ✅ Filter drivers based on the current season
        filtered_drivers = drivers_df[drivers_df["driverId"].isin(current_drivers)]

        return {
            "message": "Current season drivers fetched successfully",
            "data": filtered_drivers.to_dict(orient="records")
        }
    except Exception as e:
        return {"message": f"Error fetching drivers: {str(e)}", "data": []}









