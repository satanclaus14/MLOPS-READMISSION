import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
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

    model = XGBClassifier(
        n_estimators=params["xgboost"]["n_estimators"],
        max_depth=params["xgboost"]["max_depth"],
        learning_rate=params["xgboost"]["learning_rate"],
        subsample=params["xgboost"]["subsample"],
        colsample_bytree=params["xgboost"]["colsample_bytree"],
        reg_lambda=params["xgboost"]["reg_lambda"],
        eval_metric="logloss",
        random_state=42
    )

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params(params["xgboost"])

        model.fit(X, y)

        model_path = MODELS_DIR / "xgboost.pkl"
        joblib.dump(model, model_path)

        mlflow.xgboost.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path))

        logger.info("Trained XGBoost")

if __name__ == "__main__":
    main()
