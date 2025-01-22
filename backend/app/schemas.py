from pydantic import BaseModel

class DriverBase(BaseModel):
    name: str
    team: str
    points: float

class DriverCreate(DriverBase):
    pass

class Driver(DriverBase):
    id: int

    class Config:
        orm_mode = True
