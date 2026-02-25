import os
import json
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from src.config.paths import FEATURES_DIR

load_dotenv()

SCHEMA_FILE = FEATURES_DIR / "feature_schema.json"

def load_model():
    model_name = os.getenv("MODEL_NAME", "ReadmissionPredictionModel")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    with open(SCHEMA_FILE, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns

