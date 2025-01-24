import pandas as pd

# Load all CSVs into a dictionary for easy access
data = {
    'circuits': pd.read_csv('data/circuits.csv'),
    'constructors_results': pd.read_csv('data/constructors_results.csv'),
    'standings': pd.read_csv('data/constructors_standings.csv'),
    'constructors': pd.read_csv('data/constructors.csv'),
    'driver_standings': pd.read_csv('data/driver_standings.csv'),
    'drivers': pd.read_csv('data/drivers.csv'),
    'lap_times': pd.read_csv('data/lap_times.csv'),
    'pit_stops': pd.read_csv('data/pit_stops.csv'),
    'qualifying': pd.read_csv('data/qualifying.csv'),
    'races': pd.read_csv('data/races.csv'),
    'results': pd.read_csv('data/results.csv'),
    'seasons': pd.read_csv('data/seasons.csv'),
    'sprint_results': pd.read_csv('data/sprint_results.csv'),
    'status': pd.read_csv('data/status.csv')
}

def get_data(dataset_name):
    """Fetch data for the specified dataset."""
    if dataset_name in data:
        return data[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
