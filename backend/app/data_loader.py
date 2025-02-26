import pandas as pd

# Load CSV files into a dictionary
def load_csv_data():
    return {
        'circuits': pd.read_csv('app/data/circuits.csv', na_values=["\\N"]),
        'constructors_results': pd.read_csv('app/data/constructor_results.csv', na_values=["\\N"]),
        'standings': pd.read_csv('app/data/constructor_standings.csv', na_values=["\\N"]),
        'constructors': pd.read_csv('app/data/constructors.csv', na_values=["\\N"]),
        'driver_standings': pd.read_csv('app/data/driver_standings.csv', na_values=["\\N"]),
        'drivers': pd.read_csv('app/data/drivers.csv', na_values=["\\N"]),
        'lap_times': pd.read_csv('app/data/lap_times.csv', na_values=["\\N"]),
        'pit_stops': pd.read_csv('app/data/pit_stops.csv', na_values=["\\N"]),
        'qualifying': pd.read_csv('app/data/qualifying.csv', na_values=["\\N"]),
        'races': pd.read_csv('app/data/races.csv', na_values=["\\N"]),
        'results': pd.read_csv('app/data/results.csv', na_values=["\\N"]),
        'seasons': pd.read_csv('app/data/seasons.csv', na_values=["\\N"]),
        'sprint_results': pd.read_csv('app/data/sprint_results.csv', na_values=["\\N"]),
        'status': pd.read_csv('app/data/status.csv', na_values=["\\N"])
    }

def get_data(file_name: str):
    data = load_csv_data()
    return data.get(file_name)

