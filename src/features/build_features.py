import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.paths import CLEAN_FILE, FEATURES_DIR, FEATURE_FILE
from src.utils.logger import get_logger

logger = get_logger()

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_FILE)

    # Separate target
    y = df["readmitted"].astype(int)
    X = df.drop(columns=["readmitted"])

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Combine back for DVC
    final = X.copy()
    final["readmitted"] = y.values

    final.to_parquet(FEATURE_FILE, index=False)
    logger.info(f"Saved features file: {FEATURE_FILE}")
    logger.info(f"Shape: {final.shape}")

if __name__ == "__main__":
    main()


import json
from src.config.paths import FEATURES_DIR

SCHEMA_FILE = FEATURES_DIR / "feature_schema.json"

# Save feature column order
feature_columns = list(X.columns)

with open(SCHEMA_FILE, "w") as f:
    json.dump(feature_columns, f, indent=2)
