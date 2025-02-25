from fastapi import APIRouter
from app.data_loader import load_csv_data
from pydantic import BaseModel

# Define response model
class CircuitResponse(BaseModel):
    message: str
    data: list

router = APIRouter()

@router.get("/circuits", tags=["Data Fetching"], response_model=CircuitResponse)
def get_current_circuits():
    """Fetch the circuits used in the current season."""
    try:
        data = load_csv_data()
        circuits_df = data["circuits"]
        races_df = data["races"]

        # ✅ Get the most recent season
        current_season = races_df["year"].max()

        # ✅ Get all circuits used in the most recent season
        current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
        current_circuit_ids = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

        # ✅ Filter circuits based on the current season
        filtered_circuits = circuits_df[circuits_df["circuitId"].isin(current_circuit_ids)]

        return {
            "message": "Current season circuits fetched successfully",
            "data": filtered_circuits.to_dict(orient="records")
        }
    except Exception as e:
        return {"message": f"Error fetching circuits: {str(e)}", "data": []}







