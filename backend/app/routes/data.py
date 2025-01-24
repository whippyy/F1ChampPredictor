from fastapi import APIRouter
from app.crud import add_drivers, add_teams

router = APIRouter()

@router.get("/populate/drivers", tags=["Data Population"])
def populate_drivers():
    """Populate drivers from OpenF1 into the database."""
    result = add_drivers()
    return {"message": result}

@router.get("/populate/teams", tags=["Data Population"])
def populate_teams():
    """Populate teams from OpenF1 into the database."""
    result = add_teams()
    return {"message": result}


