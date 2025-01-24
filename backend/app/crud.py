from app.database import SessionLocal
from app.models import Driver, Team
from app.services.openf1 import fetch_drivers, fetch_teams
from sqlalchemy.orm import Session

def add_drivers():
    """Fetch drivers from OpenF1 and save to the database."""
    driver_data = fetch_drivers()
    if not driver_data:
        return "Failed to fetch driver data."
    
    db = SessionLocal()
    for driver in driver_data:
        db_driver = Driver(
            id=driver['id'],
            name=driver['name'],
            nationality=driver['nationality'],
            team=driver['constructor']
        )
        db.add(db_driver)
    db.commit()
    db.close()
    return "Drivers added successfully."

def add_teams():
    """Fetch teams from OpenF1 and save to the database."""
    team_data = fetch_teams()
    if not team_data:
        return "Failed to fetch team data."
    
    db = SessionLocal()
    for team in team_data:
        db_team = Team(
            id=team['id'],
            name=team['name'],
            nationality=team['nationality']
        )
        db.add(db_team)
    db.commit()
    db.close()
    return "Teams added successfully."

def fetch_driver_stats(db: Session, driver_id: int):
    """Fetches statistics for a specific driver."""
    return db.query(Driver).filter(Driver.id == driver_id).first()

def fetch_team_stats(db: Session, team_id: int):
    """Fetches statistics for a specific team."""
    from app.models import Team
    return db.query(Team).filter(Team.id == team_id).first()








