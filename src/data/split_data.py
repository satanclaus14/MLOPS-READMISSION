import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.config.paths import FEATURE_FILE, TRAIN_FILE, TEST_FILE
from src.utils.logger import get_logger

logger = get_logger()

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    df = pd.read_parquet(FEATURE_FILE)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["readmitted"]
    )

    train_df.to_parquet(TRAIN_FILE, index=False)
    test_df.to_parquet(TEST_FILE, index=False)

    logger.info(f"Train saved: {TRAIN_FILE} shape={train_df.shape}")
    logger.info(f"Test saved:  {TEST_FILE} shape={test_df.shape}")

if __name__ == "__main__":
    main()
