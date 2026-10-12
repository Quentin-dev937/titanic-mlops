import os
import pandas as pd

from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from src.processing import load_data, prepare_monitoring_data

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# = PROJECT_ROOT / "data" / "raw" / "titanic.csv"
REPORT_PATH = PROJECT_ROOT / "monitoring" / "reports" / "data_drift.html"

def evidently_monitoring():
    data = load_data()
    X = prepare_monitoring_data(dataframe=data)
    print(X.columns, flush=True)

    data_ref = X.sample(frac=0.7, random_state=RANDOM_STATE)

    data_prod = X.drop(data_ref.index)

    report = Report([DataDriftPreset()])

    eval_report = report.run(current_data=data_prod, reference_data=data_ref)
    eval_report.save_html(str(REPORT_PATH))

    print("completed", flush=True)
    return X




if __name__ == "__main__":
    evidently_monitoring()