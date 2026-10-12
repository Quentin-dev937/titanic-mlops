import os
import pandas as pd

from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from src.processing import load_data, prepare_monitoring_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
REPORT_PATH = PROJECT_ROOT / "monitoring" / "reports" / "data_drift.html"

def evidently_monitoring():
    data = load_data()
    X = prepare_monitoring_data(dataframe=data)
    print("completed", flush=True)
    return X




if __name__ == "__main__":
    evidently_monitoring()