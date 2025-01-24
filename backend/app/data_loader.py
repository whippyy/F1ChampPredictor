import pandas as pd

# Load all CSVs into a dictionary for easy access
data = {
    'circuits': pd.read_csv('app/data/circuits.csv'),
    'constructors_results': pd.read_csv('app/data/constructor_results.csv'),
    'standings': pd.read_csv('app/data/constructor_standings.csv'),
    'constructors': pd.read_csv('app/data/constructors.csv'),
    'driver_standings': pd.read_csv('app/data/driver_standings.csv'),
    'drivers': pd.read_csv('app/data/drivers.csv'),
    'lap_times': pd.read_csv('app/data/lap_times.csv'),
    'pit_stops': pd.read_csv('app/data/pit_stops.csv'),
    'qualifying': pd.read_csv('app/data/qualifying.csv'),
    'races': pd.read_csv('app/data/races.csv'),
    'results': pd.read_csv('app/data/results.csv'),
    'seasons': pd.read_csv('app/data/seasons.csv'),
    'sprint_results': pd.read_csv('app/data/sprint_results.csv'),
    'status': pd.read_csv('app/data/status.csv')
}

def get_data(file_name: str):
    return data.get(file_name)

