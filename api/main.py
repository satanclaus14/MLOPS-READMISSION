from fastapi import FastAPI
import pandas as pd
from api.schemas import PatientRecord, PredictionResponse
from api.model_loader import load_model
from src.monitoring.log_predictions import log_prediction

app = FastAPI(title="Readmission Prediction API")

model, feature_columns = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord):

    input_df = pd.DataFrame([record.features])

    # Align features
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    prob = float(model.predict(input_df)[0])
    pred = int(prob >= 0.5)

    log_prediction(input_df, prob, pred)

    return PredictionResponse(probability=prob, prediction=pred)

