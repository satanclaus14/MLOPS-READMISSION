import pandas as pd
from datetime import datetime
from src.config.paths import PROJECT_ROOT

PREDICTION_LOG = PROJECT_ROOT / "monitoring_logs.csv"

def log_prediction(features_df, probability, prediction):

    log_df = features_df.copy()
    log_df["probability"] = probability
    log_df["prediction"] = prediction
    log_df["timestamp"] = datetime.utcnow()

    if PREDICTION_LOG.exists():
        existing = pd.read_csv(PREDICTION_LOG)
        log_df = pd.concat([existing, log_df], ignore_index=True)

    log_df.to_csv(PREDICTION_LOG, index=False)
