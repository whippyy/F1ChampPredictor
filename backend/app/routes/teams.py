from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.crud import add_teams
from app.database import get_db
from app.schemas import ResponseMessage

router = APIRouter()

@router.get("/populate/teams", tags=["Data Population"], response_model=ResponseMessage)
def populate_teams(db: Session = Depends(get_db)):
    """Populate teams from OpenF1 into the database."""
    try:
        result = add_teams(db)
        return ResponseMessage(message=result)
    except Exception as e:
        return ResponseMessage(message=f"Error populating teams: {str(e)}")





