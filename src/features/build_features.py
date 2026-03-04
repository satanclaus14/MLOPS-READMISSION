import pandas as pd
import json
import logging
from src.config.paths import CLEAN_FILE, FEATURES_DIR, FEATURE_FILE

logging.basicConfig(level=logging.INFO)

def main():

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    df = pd.read_csv(CLEAN_FILE, low_memory=False)

    # Separate target
    y = df["readmitted"].astype(int)
    X = df.drop(columns=["readmitted"])


    SCHEMA_FILE = FEATURES_DIR / "feature_schema.json"

    feature_columns = list(X.columns)

    with open(SCHEMA_FILE, "w") as f:
        json.dump(feature_columns, f, indent=2)

    logging.info(f"Saved feature schema: {SCHEMA_FILE}")

    # -------------------------
    # Save features dataset
    # -------------------------
    final = X.copy()
    final["readmitted"] = y.values

    final.to_parquet(FEATURE_FILE, index=False)

    logging.info(f"Saved features file: {FEATURE_FILE}")
    logging.info(f"Final shape: {final.shape}")

if __name__ == "__main__":
    main()