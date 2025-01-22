from app.database import Base, engine
from app import models

# Create all the tables in the database
Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")
