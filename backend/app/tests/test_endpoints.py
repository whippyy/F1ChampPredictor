import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_get_predictions():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/predictions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)  # Replace with your expected response format

@pytest.mark.asyncio
async def test_post_predictions():
    payload = {"input_data": {"field1": "value", "field2": "value"}}
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predictions", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()  # Replace with your expected key
