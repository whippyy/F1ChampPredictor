from fastapi import APIRouter
from app.crud import add_teams
from pydantic import BaseModel

class ResponseMessage(BaseModel):
    message: str

router = APIRouter()

@router.get("/populate/teams", tags=["Data Population"], response_model=ResponseMessage)
def populate_teams():
    """Populate teams from OpenF1 into the database."""
    result = add_teams()  # Assuming this returns the number of teams added
    if result:
        return ResponseMessage(message=f"Successfully populated {result} teams.")
    return ResponseMessage(message="Failed to populate teams.")




