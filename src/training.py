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

import mlflow
import boto3
import os
from dotenv import load_dotenv

load_dotenv("./.mlflow_env")

mflow_url = os.getenv("MLFLOW_URL")


DATA_RAW_PATH = Path(r"data/raw/titanic.csv")
MODELS_PATH = Path(r"models")
TARGET_COL = "Survived"


def get_or_create_exp(name="titanic-training", artifact_location=mflow_url):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment.experiment_id is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=mflow_url)



def train():



    #remote_server_uri = "sqlite:///mlflow.db" # local for the moment
    remote_server_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(remote_server_uri)

    experiment_id = get_or_create_exp(name="titanic-training", artifact_location=mflow_url)


    print("Training", flush=True)

    data = pd.read_csv(DATA_RAW_PATH)

    data_processed = data.drop(columns=["Name", "PassengerId", "Cabin", "Ticket"])

    X = data_processed.drop(columns=[TARGET_COL])
    y = data_processed[TARGET_COL]

    num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

    columns_transformer = make_column_transformer((num_pipeline, make_column_selector(dtype_include=np.number)),
                                                 (cat_pipeline, make_column_selector(dtype_include=object)))

    
    pipeline = make_pipeline(columns_transformer, LogisticRegression())

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("model", "LogisticRegressgion")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        pipeline.fit(X_train, y_train)
        print("Model trained", flush=True)

        y_validation_pred = pipeline.predict(X_validation)
        print("Predictions completed", flush=True)

        score = round(pipeline.score(X_validation, y_validation), 2)
        print(f"Score: {score}", flush=True)

        mlflow.log_metric("validation_accuracy", score)

        mlflow.sklearn.log_model(sk_model=pipeline, name="model")


        joblib.dump(pipeline, MODELS_PATH / "titanic-model.joblib")

    

if __name__ == "__main__":
    train()
