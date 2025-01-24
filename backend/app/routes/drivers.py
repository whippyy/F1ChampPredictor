from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.crud import add_drivers
from app.database import get_db

router = APIRouter()

@router.get("/populate/drivers")
def populate_drivers(db: Session = Depends(get_db)):
    result = add_drivers(db)
    return {"message": result}


