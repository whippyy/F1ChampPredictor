from app.models import Driver, Constructor
from sqlalchemy.orm import Session
import csv

def add_drivers(db: Session):
    with open('path_to_drivers.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['number'] == '\\N':
                row['number'] = None
            
            driver = Driver(
                driver_id=row['driverId'],
                driver_ref=row['driverRef'],
                number=row['number'],
                code=row['code'],
                forename=row['forename'],
                surname=row['surname'],
                dob=row['dob'],
                nationality=row['nationality'],
                url=row['url']
            )
            db.add(driver)
        db.commit()
    return "Drivers added successfully"


def add_constructors(db: Session):
    with open('path_to_constructors.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            constructor = Constructor(
                constructor_id=row['constructorId'],
                constructor_ref=row['constructorRef'],
                name=row['name'],
                nationality=row['nationality'],
                url=row['url']
            )
            db.add(constructor)
        db.commit()
    return "Constructors added successfully"
















