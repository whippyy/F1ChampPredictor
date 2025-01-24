from sqlalchemy import Column, Integer, String
from app.database import Base

class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    nationality = Column(String)

class Constructor(Base):
    __tablename__ = "constructors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    nationality = Column(String)


