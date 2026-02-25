import pandas as pd
from sklearn.metrics import roc_auc_score
from src.config.paths import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "monitoring_logs.csv"

def main():
    df = pd.read_csv(LOG_FILE)

    if "true_label" not in df.columns:
        print("No true labels available.")
        return

    auc = roc_auc_score(df["true_label"], df["probability"])
    print("Live AUC:", auc)

if __name__ == "__main__":
    main()
