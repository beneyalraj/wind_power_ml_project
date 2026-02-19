"""
End-to-end extraction pipeline:
HDF5 → Scenario-level records → Partitioned Parquet

This script connects:
- ConfigManager
- H5StreamingReader
- PartitionedParquetWriter
"""

import logging
import sys
from datetime import datetime

from src.data.extract.h5_reader import H5StreamingReader
from src.data.extract.partitioned_writer import PartitionedParquetWriter
from src.core.config_manager import ConfigManager


# =====================================================
# Logging Setup
# =====================================================

def setup_logging():

    log_dir = "logs"
    log_file = f"{log_dir}/extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

    return log_file


# =====================================================
# Main Execution
# =====================================================

def main():

    log_file = setup_logging()
    logging.info("Starting Scenario-Level Extraction Pipeline")

    try:
        # Load and validate config/schema
        config_manager = ConfigManager()
        logging.info("Config and schema validation successful.")

        # Initialize components
        reader = H5StreamingReader()
        writer = PartitionedParquetWriter()

        logging.info("Reader and writer initialized.")

        # Execute streaming extraction
        writer.write_from_iterator(
            reader.iter_records()
        )

        logging.info("Extraction completed successfully.")
        logging.info(f"Log file saved to: {log_file}")

    except Exception as e:
        logging.exception("Extraction failed.")
        sys.exit(1)


# =====================================================

if __name__ == "__main__":
    main()
