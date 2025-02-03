from fastapi import APIRouter
from app.data_loader import get_data
from pydantic import BaseModel

# Define a response model to include a message and the data
class DriverResponse(BaseModel):
    message: str
    data: list

router = APIRouter()

@router.get("/drivers", tags=["Data Fetching"], response_model=DriverResponse)
async def get_drivers():
    try:
        drivers_data = get_data('drivers')
        if drivers_data is not None:
            return DriverResponse(message="Drivers data fetched successfully", data=drivers_data.to_dict(orient='records'))
        return DriverResponse(message="No data found for drivers.", data=[])
    except Exception as e:
        return DriverResponse(message=f"Error fetching drivers data: {str(e)}", data=[])

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Driver

router = APIRouter()

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fetch all drivers
@router.get("/drivers")
def get_all_drivers(db: Session = Depends(get_db)):
    drivers = db.query(Driver).all()
    return drivers

# Fetch a specific driver by ID
@router.get("/drivers/{driver_id}")
def get_driver_by_id(driver_id: int, db: Session = Depends(get_db)):
    driver = db.query(Driver).filter(Driver.driver_id == driver_id).first()
    if not driver:
        raise HTTPException(status_code=404, detail="Driver not found")
    return driver








