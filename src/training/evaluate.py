import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from src.config.paths import TEST_FILE, MODELS_DIR, REPORTS_DIR, METRICS_FILE
from src.utils.logger import get_logger

logger = get_logger()

def eval_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y, probs)),
        "f1": float(f1_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall": float(recall_score(y, preds))
    }

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TEST_FILE)
    y = df["readmitted"]
    X = df.drop(columns=["readmitted"])

    metrics = {}

    for name, file in [
        ("log_reg", MODELS_DIR / "log_reg.pkl"),
        ("random_forest", MODELS_DIR / "random_forest.pkl"),
        ("xgboost", MODELS_DIR / "xgboost.pkl"),
        ]:
        model = joblib.load(file)

    # 🔥 Align test features to training features
        if hasattr(model, "feature_names_in_"):
            X_aligned = X.reindex(columns=model.feature_names_in_, fill_value=0)
        else:
            X_aligned = X

        metrics[name] = eval_model(model, X_aligned, y)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics: {METRICS_FILE}")
    logger.info(metrics)

if __name__ == "__main__":
    main()
