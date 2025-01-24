from fastapi import APIRouter
from app.data_loader import get_data  # Assuming you have a data_loader function for fetching data

router = APIRouter()

@router.get("/circuits")
async def get_circuits():
    circuits_data = get_data('circuits')
    return circuits_data.to_dict(orient='records')  # Convert DataFrame to list of dicts





