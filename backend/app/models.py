# app/models.py
from sqlalchemy import Column, Integer, String, Float
from app.database import Base

class Driver(Base):
    __tablename__ = 'drivers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    team = Column(String)
    points = Column(Float)

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    total_points = Column(Float)
    total_wins = Column(Integer)

