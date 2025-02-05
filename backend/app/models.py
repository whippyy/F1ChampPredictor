from sqlalchemy import Column, Integer, String, Date
from app.database import Base

class Driver(Base):
    __tablename__ = "drivers"
    driver_id = Column(Integer, primary_key=True, index=True)
    driver_ref = Column(String, index=True)
    number = Column(String, nullable=True)
    code = Column(String, nullable=True)
    forename = Column(String, nullable=False)
    surname = Column(String, nullable=False)
    dob = Column(Date, nullable=False)
    nationality = Column(String, nullable=False)
    url = Column(String, nullable=False)

    def to_dict(self):
        return {
            "driverId": self.driver_id,
            "driverRef": self.driver_ref,
            "number": self.number,
            "code": self.code,
            "forename": self.forename,
            "surname": self.surname,
            "dob": str(self.dob),
            "nationality": self.nationality,
            "url": self.url,
        }

class Constructor(Base):
    __tablename__ = "constructors"
    constructor_id = Column(Integer, primary_key=True, index=True)
    constructor_ref = Column(String, nullable=False)
    name = Column(String, nullable=False)
    nationality = Column(String, nullable=False)
    url = Column(String, nullable=False)





