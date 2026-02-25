import pandas as pd
from src.config.paths import RAW_FILE, PROCESSED_DIR, CLEAN_FILE
from src.utils.logger import get_logger

logger = get_logger()

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    # Standard cleaning
    df.columns = [c.strip().lower() for c in df.columns]

    # Replace '?' with NaN
    df = df.replace("?", pd.NA)

    # Drop duplicates if any
    df = df.drop_duplicates()

    # Drop columns that are too ID-like / leakage-like
    drop_cols = ["encounter_id", "patient_nbr"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target conversion
    # readmitted values: "<30", ">30", "NO"
    df["readmitted"] = df["readmitted"].map(
        {"<30": 1, ">30": 0, "NO": 0}
    )

    # Drop rows where target is missing
    df = df.dropna(subset=["readmitted"])

    df.to_csv(CLEAN_FILE, index=False)
    logger.info(f"Saved cleaned data: {CLEAN_FILE}")
    logger.info(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
