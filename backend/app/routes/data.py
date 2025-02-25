from fastapi import APIRouter
from app.data_loader import get_data, load_csv_data
from pydantic import BaseModel

# Define a response model to include a message and the data
class CircuitResponse(BaseModel):
    message: str
    data: list

router = APIRouter()

@router.get("/circuits", tags=["Data Fetching"], response_model=CircuitResponse)
async def get_circuits():
    try:
        circuits_data = get_data('circuits')
        if circuits_data is not None:
            return CircuitResponse(message="Circuits data fetched successfully", data=circuits_data.to_dict(orient='records'))
        return CircuitResponse(message="No data found for circuits.", data=[])
    except Exception as e:
        return CircuitResponse(message=f"Error fetching circuits data: {str(e)}", data=[])

@router.get("/circuits")
def get_current_circuits():
    """Fetch the circuits used in the current season."""
    data = load_csv_data()
    circuits_df = data["circuits"]
    races_df = data["races"]

    # Get the most recent season
    current_season = races_df["year"].max()

    # Get all circuits used in the most recent season
    current_season_race_ids = races_df[races_df["year"] == current_season]["raceId"]
    current_circuit_ids = races_df[races_df["raceId"].isin(current_season_race_ids)]["circuitId"].unique()

    filtered_circuits = circuits_df[circuits_df["circuitId"].isin(current_circuit_ids)]

    return {
        "message": "Current season circuits fetched successfully",
        "data": filtered_circuits.to_dict(orient="records")
    }






