from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.crud import add_teams
from app.database import get_db

router = APIRouter()

@router.get("/populate/teams")
def populate_teams(db: Session = Depends(get_db)):
    result = add_teams(db)
    return {"message": result}


