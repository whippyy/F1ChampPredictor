from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.crud import add_drivers_with_message
from app.database import get_db
from app.schemas import ResponseMessage

router = APIRouter()

@router.get("/populate/drivers", tags=["Data Population"], response_model=ResponseMessage)
def populate_drivers(db: Session = Depends(get_db)):
    """Populate drivers from OpenF1 into the database."""
    result = add_drivers_with_message(db)  # Get the result as a message
    return result  # Response will be formatted with message




