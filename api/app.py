import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH",
                            PROJECT_ROOT / "models" / "titanic-model.joblib"))

model = joblib.load(MODEL_PATH)


app = FastAPI(title="titanic predictions API", version="1.0.0")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Passenger):

    df = pd.DataFrame([data.model_dump()])

    predictions = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {"survived": int(predictions),
            "probability": float(proba)}
