import requests
from typing import Optional

BASE_URL = "https://api.openf1.org/v1"
API_KEY = "your_api_key_here"  # Replace with your OpenF1 API key

headers = {"Authorization": f"Bearer {API_KEY}"}

def fetch_drivers() -> Optional[dict]:
    """Fetches driver data from OpenF1."""
    url = f"{BASE_URL}/drivers"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch drivers: {response.status_code}")
        return None

def fetch_teams() -> Optional[dict]:
    """Fetches team data from OpenF1."""
    url = f"{BASE_URL}/constructors"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch teams: {response.status_code}")
        return None

def fetch_race_results(year: int, round: int) -> Optional[dict]:
    """Fetches race results for a specific year and round."""
    url = f"{BASE_URL}/results/{year}/{round}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch race results: {response.status_code}")
        return None
