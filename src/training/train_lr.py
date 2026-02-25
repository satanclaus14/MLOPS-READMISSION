import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from src.config.paths import TRAIN_FILE, MODELS_DIR
from src.utils.logger import get_logger
from src.utils.mlflow_utils import setup_mlflow

logger = get_logger()

def main():
    setup_mlflow()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    df = pd.read_parquet(TRAIN_FILE)
    y = df["readmitted"]
    X = df.drop(columns=["readmitted"])

    model = LogisticRegression(
        C=params["log_reg"]["C"],
        max_iter=params["log_reg"]["max_iter"]
    )

    with mlflow.start_run(run_name="log_reg"):
        mlflow.log_params(params["log_reg"])

        model.fit(X, y)

        model_path = MODELS_DIR / "log_reg.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path))

        logger.info("Trained Logistic Regression")

if __name__ == "__main__":
    main()
