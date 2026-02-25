from src.config.paths import RAW_DIR, RAW_FILE
from src.utils.logger import get_logger

logger = get_logger()

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        logger.info(f"Raw dataset already exists at: {RAW_FILE}")
        return

    raise FileNotFoundError(
        f"{RAW_FILE} not found.\n"
        "Download the Diabetes 130-US hospitals dataset and place it as:\n"
        "data/raw/readmission.csv"
    )

if __name__ == "__main__":
    main()
