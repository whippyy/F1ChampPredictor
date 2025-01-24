from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.crud import fetch_driver_stats, fetch_team_stats
from app.database import get_db  # Ensure you import get_db

router = APIRouter()

@router.get("/{driver_id}/features")
def get_driver_features(driver_id: int, db: Session = Depends(get_db)):
    """Fetch the combined feature vector for a given driver."""
    driver_stats = fetch_driver_stats(driver_id, db)
    team_stats = fetch_team_stats(driver_id, db)  # Example, adjust as needed

    if driver_stats and team_stats:
        feature_vector = [
            driver_stats['total_points'],
            driver_stats['total_races'],
            team_stats['total_points'],
            team_stats['total_wins']
        ]
        return {"feature_vector": feature_vector}
    return {"error": "Driver or team data not found"}

