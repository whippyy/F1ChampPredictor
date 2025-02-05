from sqlalchemy import Column, Integer, String, Date
from app.database import Base

class Driver(Base):
    __tablename__ = "drivers"
    driver_id = Column(Integer, primary_key=True, index=True)
    driver_ref = Column(String, index=True)
    number = Column(String)
    code = Column(String)
    forename = Column(String)
    surname = Column(String)
    dob = Column(Date)
    nationality = Column(String)
    url = Column(String)

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

    id = Column(Integer, primary_key=True, index=True)
    constructor_id = Column(Integer, unique=True, index=True)
    constructor_ref = Column(String)
    name = Column(String)
    nationality = Column(String)
    url = Column(String)





