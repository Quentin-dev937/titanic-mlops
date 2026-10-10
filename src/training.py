import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from pathlib import Path
import joblib


import boto3
import psycopg2
import shap
import matplotlib
import os
from dotenv import load_dotenv



DATA_RAW_PATH = Path(r"data/raw/titanic.csv")
MODELS_PATH = Path(r"models")
TARGET_COL = "Survived"

THRESHOLD = 0.80



def train():


    print("Training", flush=True)

    data = pd.read_csv(DATA_RAW_PATH)

    data_processed = data.drop(columns=["Name", "PassengerId", "Cabin", "Ticket"])

    X = data_processed.drop(columns=[TARGET_COL])
    y = data_processed[TARGET_COL]

    num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

    columns_transformer = make_column_transformer((num_pipeline, make_column_selector(dtype_include=np.number)),
                                                 (cat_pipeline, make_column_selector(dtype_include=object)))

    
    pipeline = make_pipeline(columns_transformer, LogisticRegression(random_state=42, C=1))

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)


    pipeline.fit(X_train, y_train)
    print("Model trained", flush=True)

    train_score = round(pipeline.score(X_train, y_train), 2)
    print(f"train_accuracy: {train_score}", flush=True)

    y_validation_pred = pipeline.predict(X_validation)
    print("Predictions completed", flush=True)

    validation_score = round(pipeline.score(X_validation, y_validation), 2)
    print(f"validation_score: {validation_score}", flush=True)

    joblib.dump(pipeline, MODELS_PATH / "titanic-model.joblib")

    

if __name__ == "__main__":
    train()
