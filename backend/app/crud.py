from sqlalchemy.orm import Session
from app import models
import pandas as pd 

# Function to add drivers to the database
def add_drivers(db: Session):
    # Example: Insert data from CSV into the database
    drivers_data = pd.read_csv('app/data/drivers.csv')
    for index, row in drivers_data.iterrows():
        db.add(models.Driver(name=row['name'], nationality=row['nationality']))
    db.commit()
    return "Drivers added successfully."

# Function to add teams (constructors)
def add_teams(db: Session):
    constructors_data = pd.read_csv('app/data/constructors.csv')
    for index, row in constructors_data.iterrows():
        db.add(models.Constructor(name=row['name'], nationality=row['nationality']))
    db.commit()
    return "Teams added successfully."

# Function to fetch driver stats from the database
def fetch_driver_stats(driver_id: int, db: Session):
    return db.query(models.Driver).filter(models.Driver.id == driver_id).first()

# Function to fetch team stats from the database
def fetch_team_stats(driver_id: int, db: Session):
    driver = db.query(models.Driver).filter(models.Driver.id == driver_id).first()
    return db.query(models.Constructor).filter(models.Constructor.id == driver.team_id).first()










