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









