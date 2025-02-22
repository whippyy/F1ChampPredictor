from fastapi import APIRouter
from app.data_loader import get_data
from pydantic import BaseModel

# Define a response model to include a message and the data
class TeamResponse(BaseModel):
    message: str
    data: list

router = APIRouter()

@router.get("/teams", tags=["Data Fetching"], response_model=TeamResponse)
async def get_teams():
    try:
        teams_data = get_data('constructors')
        if teams_data is not None:
            return TeamResponse(message="Teams data fetched ", data=teams_data.to_dict(orient='records'))
        return TeamResponse(message="No data found for teams.", data=[])
    except Exception as e:
        return TeamResponse(message=f"Error fetching teams data: {str(e)}", data=[])





