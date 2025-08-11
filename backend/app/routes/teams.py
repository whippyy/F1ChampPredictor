from fastapi import APIRouter, HTTPException
from app.data_loader import f1_data
from app.schemas import Team
from typing import List

router = APIRouter()

@router.get("/teams", response_model=List[Team], tags=["Data Fetching"])
def get_current_teams() -> List[Team]:
    """Fetch all teams (constructors) that participated in the current season."""
    try:
        constructors_df = f1_data.data["constructors"]
        current_teams_ids = f1_data.current_constructor_ids

        filtered_teams = constructors_df[constructors_df["constructorId"].isin(current_teams_ids)]

        return filtered_teams.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")
