from app.database import db_session
from app.models import Driver  # Assuming you have a Driver model defined

def add_driver_from_api(driver_data):
    for driver in driver_data:
        db_driver = Driver(
            id=driver['id'],
            name=driver['name'],
            team=driver['team'],
            nationality=driver['nationality'],
            # Add other fields as necessary, based on the data returned by the API
        )
        db_session.add(db_driver)
    
    db_session.commit()

# Example functions to fetch driver stats and team stats
def fetch_driver_stats(driver_id: int):
    return db_session.query(Driver).filter(Driver.id == driver_id).first()

def fetch_team_stats(team_id: int):
    return db_session.query(Team).filter(Team.id == team_id).first()





