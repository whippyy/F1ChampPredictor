# app/models.py
from sqlalchemy import Column, Integer, String, Date
from app.database import Base  # Assuming you're using Base from your database setup

class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(Integer, unique=True, index=True)
    driver_ref = Column(String)
    number = Column(Integer)
    code = Column(String, unique=True, index=True)
    forename = Column(String)
    surname = Column(String)
    dob = Column(Date)
    nationality = Column(String)
    url = Column(String)


class Constructor(Base):
    __tablename__ = "constructors"

    id = Column(Integer, primary_key=True, index=True)
    constructor_id = Column(Integer, unique=True, index=True)
    constructor_ref = Column(String)
    name = Column(String)
    nationality = Column(String)
    url = Column(String)




