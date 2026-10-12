import os
import pandas as pd

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "titanic.csv"


COLUMNS_TO_DROP = ["Name", "PassengerId", "Cabin", "Ticket"]
TARGET_COL = "Survived"



def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def split_X_y(dataframe):
    X = dataframe.drop(columns=[TARGET_COL, *COLUMNS_TO_DROP], errors="ignore")
    y = dataframe[TARGET_COL]
    return X, y

def prepare_monitoring_data(dataframe):
    return dataframe.drop(columns=[TARGET_COL, *COLUMNS_TO_DROP], errors="ignore")

def save_data(dataframe, path):
    dataframe.to_csv(path, index=False)