import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
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

    model = RandomForestClassifier(
        n_estimators=params["random_forest"]["n_estimators"],
        max_depth=params["random_forest"]["max_depth"],
        min_samples_split=params["random_forest"]["min_samples_split"],
        min_samples_leaf=params["random_forest"]["min_samples_leaf"],
        n_jobs=-1,
        random_state=42
    )

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params(params["random_forest"])

        model.fit(X, y)

        model_path = MODELS_DIR / "random_forest.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_path))

        logger.info("Trained Random Forest")

if __name__ == "__main__":
    main()
