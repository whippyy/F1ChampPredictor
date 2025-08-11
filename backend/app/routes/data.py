from fastapi import APIRouter, HTTPException
from app.data_loader import f1_data
from app.schemas import Circuit
from typing import List

router = APIRouter()

@router.get("/circuits", response_model=List[Circuit], tags=["Data Fetching"])
def get_current_circuits() -> List[Circuit]:
    """Fetch the circuits used in the current season."""
    try:
        circuits_df = f1_data.data["circuits"]
        current_circuit_ids = f1_data.current_circuit_ids

        # ✅ Filter circuits based on the current season
        filtered_circuits = circuits_df[circuits_df["circuitId"].isin(current_circuit_ids)]

        return filtered_circuits.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching circuits: {str(e)}")
