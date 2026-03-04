import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "readmission-prediction"))
