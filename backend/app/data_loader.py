import pandas as pd
import joblib
from pathlib import Path

# Singleton pattern to load data once
class F1DataLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_data()
        return cls._instance
    
    def _load_data(self):
        data_dir = Path(__file__).parent.parent / "data"
        self.data = {
            "drivers": pd.read_csv(data_dir / "drivers.csv"),
            "circuits": pd.read_csv(data_dir / "circuits.csv"),
            "races": pd.read_csv(data_dir / "races.csv"),
            "results": pd.read_csv(data_dir / "results.csv"),
            # ... load all other CSV files
        }
        
        # Pre-compute useful views
        self.current_year = 2024
        self.current_races = self.data["races"][self.data["races"]["year"] == self.current_year]
        
    def get_data(self):
        return self.data
    
    def get_current_races(self):
        return self.current_races
    
    # Add other convenience methods as needed

# Global instance
f1_data = F1DataLoader()