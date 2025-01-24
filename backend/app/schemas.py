from pydantic import BaseModel

class DriverBase(BaseModel):
    name: str
    nationality: str

class DriverCreate(DriverBase):
    pass

class Driver(DriverBase):
    id: int

    class Config:
        orm_mode = True  # Tells Pydantic to treat SQLAlchemy models as dictionaries




