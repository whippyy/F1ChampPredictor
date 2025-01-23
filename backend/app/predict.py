# app/predict.py
from app.crud import fetch_driver_stats, fetch_team_stats

def prepare_feature_vector(driver_id: int, team_id: int):
    """Prepare the feature vector for model prediction."""
    driver_stats = fetch_driver_stats(driver_id)
    team_stats = fetch_team_stats(team_id)
    
    if driver_stats and team_stats:
        feature_vector = [
            driver_stats['total_points'],
            driver_stats['total_races'],
            team_stats['total_points'],
            team_stats['total_wins']
        ]
        return feature_vector
    return None

