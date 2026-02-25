import pandas as pd
import numpy as np
import json
from src.config.paths import TRAIN_FILE, PROJECT_ROOT

PREDICTION_LOG = PROJECT_ROOT / "monitoring_logs.csv"
DRIFT_REPORT = PROJECT_ROOT / "drift_report.json"

def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    psi = 0
    for i in range(len(expected_perc)-1):
        e = ((expected >= expected_perc[i]) & (expected < expected_perc[i+1])).mean()
        a = ((actual >= actual_perc[i]) & (actual < actual_perc[i+1])).mean()

        if e > 0 and a > 0:
            psi += (a - e) * np.log(a / e)

    return psi

def main():
    train_df = pd.read_parquet(TRAIN_FILE)
    pred_df = pd.read_csv(PREDICTION_LOG)

    drift_scores = {}

    for col in train_df.columns:
        if col == "readmitted":
            continue
        if col not in pred_df.columns:
            continue

        psi = calculate_psi(train_df[col], pred_df[col])
        drift_scores[col] = psi

    avg_psi = float(np.mean(list(drift_scores.values())))

    result = {
        "avg_psi": avg_psi,
        "feature_drift": drift_scores
    }

    with open(DRIFT_REPORT, "w") as f:
        json.dump(result, f, indent=2)

    print("Drift report saved:", result)

if __name__ == "__main__":
    main()
