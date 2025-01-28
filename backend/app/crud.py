from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app import models
import pandas as pd
from app.schemas import ResponseMessage

def add_drivers(db: Session):
    try:
        drivers_data = pd.read_csv('app/data/drivers.csv')
        for index, row in drivers_data.iterrows():
            db.add(models.Driver(
                code=row['code'],
                team_id=row['team_id'],
                nationality=row['nationality'],
                full_name=row['full_name'],
                dob=row['dob'],
                url=row['url']
            ))
        db.commit()
        return "Drivers added successfully."
    except SQLAlchemyError as e:
        db.rollback()
        return f"Error adding drivers: {e}"

def add_teams(db: Session):
    try:
        constructors_data = pd.read_csv('app/data/constructors.csv')
        for index, row in constructors_data.iterrows():
            db.add(models.Constructor(
                name=row['name'],
                nationality=row['nationality']
            ))
        db.commit()
        return "Teams added successfully."
    except SQLAlchemyError as e:
        db.rollback()
        return f"Error adding teams: {e}"

def fetch_driver_stats(driver_id: int, db: Session):
    return db.query(models.Driver).filter(models.Driver.id == driver_id).first()

def fetch_team_stats(driver_id: int, db: Session):
    driver = db.query(models.Driver).filter(models.Driver.id == driver_id).first()
    if driver:
        return db.query(models.Constructor).filter(models.Constructor.id == driver.team_id).first()
    return None

def add_drivers_with_message(db: Session):
    result = add_drivers(db)
    return ResponseMessage(message=result)










