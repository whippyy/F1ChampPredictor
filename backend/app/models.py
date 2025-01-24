from sqlalchemy import Column, Integer, String
from app.database import Base

class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, index=True)
    team_id = Column(Integer, index=True)
    nationality = Column(String)
    full_name = Column(String)
    dob = Column(Date)
    url = Column(String)

    # Define the relationship if you have a team model
    team = relationship("Constructor", back_populates="drivers")


class Constructor(Base):
    __tablename__ = "constructors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    nationality = Column(String)


