import pandas as pd
import joblib
import logging
from pathlib import Path

class F1DataLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_data()
        return cls._instance
    
    def _load_data(self):
        # Point to the correct data directory
        data_dir = Path(__file__).parent / "data"
        logging.info(f"Loading data from: {data_dir}")
        self.data = {
            "drivers": pd.read_csv(data_dir / "drivers.csv"),
            "circuits": pd.read_csv(data_dir / "circuits.csv"),
            "races": pd.read_csv(data_dir / "races.csv"),
            "results": pd.read_csv(data_dir / "results.csv"),
            "lap_times": pd.read_csv(data_dir / "lap_times.csv"),
            "pit_stops": pd.read_csv(data_dir / "pit_stops.csv"),
            "qualifying": pd.read_csv(data_dir / "qualifying.csv"),
            "driver_standings": pd.read_csv(data_dir / "driver_standings.csv"),
            "standings": pd.read_csv(data_dir / "constructor_standings.csv"),
            "constructors": pd.read_csv(data_dir / "constructors.csv")
        }
        
        # Pre-compute useful data for the current season
        self.current_year = self.data["races"]["year"].max()
        
        self.current_races_df = self.data["races"][self.data["races"]["year"] == self.current_year]
        self.current_race_ids = self.current_races_df["raceId"].unique()
        
        current_results_df = self.data["results"][self.data["results"]["raceId"].isin(self.current_race_ids)]
        
        self.current_driver_ids = current_results_df["driverId"].unique()
        self.current_constructor_ids = current_results_df["constructorId"].unique()
        self.current_circuit_ids = self.current_races_df["circuitId"].unique()

        # Create a mapping of driverId to their latest constructorId for the current season
        last_driver_results = current_results_df.sort_values("raceId").drop_duplicates("driverId", keep="last")
        self.driver_team_map = pd.Series(
            last_driver_results.constructorId.values, 
            index=last_driver_results.driverId
        ).to_dict()

        logging.info(f"Data loaded for current year: {self.current_year}")

# Global instance
f1_data = F1DataLoader()

# Legacy function for backward compatibility
def load_csv_data():
    """Maintain this for existing imports"""
    return f1_data.data