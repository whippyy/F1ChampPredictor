from sqlalchemy.orm import Session
from app.crud import fetch_driver_stats, fetch_team_stats

def prepare_feature_vector(db: Session, driver_id: int, team_id: int):
    """Prepare the feature vector for prediction."""
    driver_stats = fetch_driver_stats(db, driver_id)
    team_stats = fetch_team_stats(db, team_id)
    # Process driver_stats and team_stats into a feature vector
    feature_vector = {
        "driver_wins": driver_stats.wins,
        "team_points": team_stats.points,
        # Add other relevant features
    }
    return feature_vector

