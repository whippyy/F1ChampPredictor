import pandas as pd
import joblib
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
        print(f"Loading data from: {data_dir}")
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
        # Debug: Print loaded driver IDs
        print(f"Loaded {len(self.data['drivers'])} drivers")
        print("Sample drivers:", self.data['drivers']['driverId'].head().tolist())
        
        # Pre-compute useful views
        self.current_year = 2024
        self.current_races = self.data["races"][self.data["races"]["year"] == self.current_year]

# Global instance
f1_data = F1DataLoader()

# Legacy function for backward compatibility
def load_csv_data():
    """Maintain this for existing imports"""
    return f1_data.data