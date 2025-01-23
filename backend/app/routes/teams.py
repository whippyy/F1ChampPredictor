from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_teams():
    return {"message": "Teams endpoint is working!"}
