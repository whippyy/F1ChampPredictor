# app/crud.py
from sqlalchemy.orm import Session
from app import models, schemas
from app.database import db_session
from app.models import Driver, Team

def create_driver(db: Session, driver: schemas.DriverCreate):
    db_driver = models.Driver(**driver.dict())
    db.add(db_driver)
    db.commit()
    db.refresh(db_driver)
    return db_driver

def get_driver(db: Session, driver_id: int):
    return db.query(models.Driver).filter(models.Driver.id == driver_id).first()

def fetch_driver_stats(driver_id: int):
    """Fetch historical performance data for a driver."""
    driver = db_session.query(Driver).filter(Driver.id == driver_id).first()
    if driver:
        return {
            "total_points": driver.total_points,
            "total_races": driver.total_races,
            # You can return other relevant stats
        }
    return None

def fetch_team_stats(team_id: int):
    """Fetch historical performance data for a team."""
    team = db_session.query(Team).filter(Team.id == team_id).first()
    if team:
        return {
            "total_points": team.total_points,
            "total_wins": team.total_wins,
            # You can return other relevant stats
        }
    return None

