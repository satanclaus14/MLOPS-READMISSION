import os
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from dotenv import load_dotenv
from src.config.paths import BEST_MODEL_FILE, MODELS_DIR
from src.utils.logger import get_logger
from src.utils.mlflow_utils import setup_mlflow

logger = get_logger()

load_dotenv()

def main():
    setup_mlflow()

    model_name = os.getenv("MODEL_NAME", "ReadmissionPredictionModel")

    with open(BEST_MODEL_FILE, "r") as f:
        best = json.load(f)

    if not best["passed_threshold"]:
        raise ValueError("Best model did not pass threshold. Not registering.")

    best_model = best["best_model"]

    model_path = MODELS_DIR / f"{best_model}.pkl"
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="register_best_model"):
        if best_model == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=model_name)
        else:
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    logger.info(f"Registered model '{best_model}' as '{model_name}' in MLflow")

if __name__ == "__main__":
    main()
