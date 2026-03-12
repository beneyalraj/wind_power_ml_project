import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

SPEC_PATH = Path("configs/feature_spec.yaml")
INPUT_DIR = Path("data/interim/splits")
OUTPUT_DIR = Path("data/features")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------
# Logging Setup
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "feature_engineering.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Load Feature Specification
# ---------------------------------------------------

def load_feature_spec():
    try:
        with open(SPEC_PATH) as f:
            logger.info(f"Loading feature specification from {SPEC_PATH}")
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load feature specification: {e}")
        raise


# ---------------------------------------------------
# Validate Input Schema
# ---------------------------------------------------

def validate_schema(df, identifiers, numerical, target):

    expected_columns = identifiers + numerical + [target]

    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ---------------------------------------------------
# Build Derived Features
# ---------------------------------------------------

def build_features(df, spec):

    identifiers = spec["identifiers"]
    numerical = spec["numerical_features"]
    derived = spec.get("derived_features", [])
    target = spec["target"]

    validate_schema(df, identifiers, numerical, target)

    df = df[identifiers + numerical + [target]].copy()

    supported_features = {
        "wind_direction_sin",
        "wind_direction_cos",
        "wind_speed_squared"
    }

    for feature in derived:

        if feature not in supported_features:
            raise ValueError(f"Unsupported derived feature: {feature}")

        if feature == "wind_direction_sin":
            df["wind_direction_sin"] = np.sin(
                np.radians(df["wind_direction"])
            )

        elif feature == "wind_direction_cos":
            df["wind_direction_cos"] = np.cos(
                np.radians(df["wind_direction"])
            )

        elif feature == "wind_speed_squared":
            df["wind_speed_squared"] = df["wind_speed"] ** 2

    return df


# ---------------------------------------------------
# Process Dataset Split
# ---------------------------------------------------

def process_split(split_name, spec):

    input_path = INPUT_DIR / split_name
    output_path = OUTPUT_DIR / split_name

    output_path.mkdir(parents=True, exist_ok=True)

    identifiers = spec["identifiers"]
    numerical = spec["numerical_features"]
    target = spec["target"]

    parquet_files = list(input_path.glob("*.parquet"))

    logger.info(
        f"Processing {split_name} split: {len(parquet_files)} files found"
    )

    for file in parquet_files:

        try:

            df = pd.read_parquet(
                file,
                columns=identifiers + numerical + [target]
            )

            df = build_features(df, spec)

            output_file = output_path / file.name

            df.to_parquet(output_file)

            logger.info(f"Processed file: {file.name}")

        except Exception as e:

            logger.error(
                f"Error processing file {file.name}: {e}"
            )

            raise


# ---------------------------------------------------
# Main Pipeline Entry
# ---------------------------------------------------

def main():

    logger.info("Starting feature extraction pipeline")

    spec = load_feature_spec()

    for split in ["train", "validation", "test"]:

        process_split(split, spec)

    logger.info("Feature extraction pipeline completed successfully")


# ---------------------------------------------------

if __name__ == "__main__":
    main()