import json
import yaml
from src.config.paths import METRICS_FILE, BEST_MODEL_FILE, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger()

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    metric_name = params["training"]["metric"]
    threshold = params["training"]["min_auc_threshold"]

    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)

    best_model = None
    best_score = -1

    for model_name, vals in metrics.items():
        score = vals[metric_name]
        if score > best_score:
            best_score = score
            best_model = model_name

    result = {
        "best_model": best_model,
        "best_score": best_score,
        "metric": metric_name,
        "threshold": threshold,
        "passed_threshold": best_score >= threshold
    }

    with open(BEST_MODEL_FILE, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Best model selection: {result}")

if __name__ == "__main__":
    main()
