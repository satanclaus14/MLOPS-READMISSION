from src.config.paths import RAW_DIR, RAW_FILE
import logging

logging.basicConfig(level=logging.INFO)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        logging.info(f"Raw dataset already exists at: {RAW_FILE}")
    else:
        raise FileNotFoundError(
            f"{RAW_FILE} not found.\n"
            f"Download the Diabetes 130-US hospitals dataset and place it as:\n"
            f"data/raw/diabetic_data.csv"
        )

if __name__ == "__main__":
    main()