"""
Partitioned Parquet writer for scenario-level dataset.

Responsibilities:
- Batch buffering
- Partitioned writes (layout_id)
- Deterministic file naming
- Idempotent behavior
"""

from pathlib import Path
from typing import Iterator, Dict, Any, List
import pandas as pd
import logging

from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class PartitionedParquetWriter:

    def __init__(self, config_path: str = "params.yaml"):

        self.cm = ConfigManager(config_path)
        self.config = self.cm.get_extraction_config()
        self.schema = self.cm.get_schema()

        self.output_path = Path(self.config["output_path"])
        self.rows_per_batch = self.config["rows_per_batch"]
        self.partition_keys = self.config["partition_by"]

        parquet_config = self.config.get("parquet", {})
        self.compression = parquet_config.get("compression", "snappy")
        self.row_group_size = parquet_config.get("row_group_size", 10000)

        self.output_path.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Public Write Entry
    # =====================================================

    def write_from_iterator(
        self,
        record_iterator: Iterator[Dict[str, Any]]
    ):

        buffer: List[Dict[str, Any]] = []

        for record in record_iterator:
            buffer.append(record)

            if len(buffer) >= self.rows_per_batch:
                self._flush_buffer(buffer)
                buffer.clear()

        # Flush remaining
        if buffer:
            self._flush_buffer(buffer)

        logger.info("✅ Parquet writing complete.")

    # =====================================================
    # Flush Logic
    # =====================================================

    def _flush_buffer(self, buffer: List[Dict[str, Any]]):

        df = pd.DataFrame(buffer)

        # Ensure consistent column order
        df = df[self._get_ordered_columns()]

        # Partition by layout_id (scenario-level)
        grouped = df.groupby(self.partition_keys)

        for partition_values, partition_df in grouped:

            if not isinstance(partition_values, tuple):
                partition_values = (partition_values,)

            partition_path = self._build_partition_path(partition_values)

            partition_path.mkdir(parents=True, exist_ok=True)

            file_path = self._next_file_path(partition_path)

            partition_df.to_parquet(
                file_path,
                index=False,
                compression=self.compression,
                engine="pyarrow"
            )

    # =====================================================
    # Helpers
    # =====================================================

    def _build_partition_path(self, partition_values: tuple) -> Path:

        partition_path = self.output_path

        for key, value in zip(self.partition_keys, partition_values):
            partition_path = partition_path / f"{key}={value}"

        return partition_path

    def _next_file_path(self, partition_path: Path) -> Path:

        existing_files = sorted(partition_path.glob("part-*.parquet"))

        if not existing_files:
            next_index = 0
        else:
            last_file = existing_files[-1].stem
            last_index = int(last_file.split("-")[1])
            next_index = last_index + 1

        return partition_path / f"part-{next_index:05d}.parquet"

    def _get_ordered_columns(self) -> List[str]:

        identifiers = list(self.schema["identifiers"].keys())
        features = list(self.schema["features"].keys())
        target = list(self.schema["target"].keys())

        return identifiers + features + target
