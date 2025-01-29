import requests

# Example URL for fetching driver data
url = "https://api.openf1.org/2024/drivers"

def fetch_driver_data():
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

