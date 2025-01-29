from fastapi import APIRouter
from app.data_loader import get_data
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






