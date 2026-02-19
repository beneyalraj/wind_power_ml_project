"""
Partition-aware, schema-driven validation
for scenario-level Parquet dataset.
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from src.core.config_manager import ConfigManager


# ============================================================
# Core Validation Logic
# ============================================================

def validate():

    cm = ConfigManager()
    schema = cm.get_schema()

    base_path = Path("data/interim/scenario_dataset_v1")
    files = sorted(base_path.glob("layout_id=*/part-*.parquet"))

    if not files:
        raise ValueError("No Parquet files found in data/interim")

    primary_keys = schema["primary_key"]
    identifiers = schema["identifiers"]
    features = schema["features"]
    target = schema["target"]

    expected_columns = (
        list(identifiers.keys())
        + list(features.keys())
        + list(target.keys())
    )

    seen_keys = set()

    metrics = {
        "total_rows": 0,
        "duplicate_primary_keys": 0,
        "null_violations": 0,
        "range_violations": 0,
        "dtype_violations": 0,
        "missing_columns": 0,
        "partitions_checked": 0
    }

    # ========================================================
    # Partition-wise validation
    # ========================================================

    for file in files:

        df = pd.read_parquet(file)
        metrics["partitions_checked"] += 1
        metrics["total_rows"] += len(df)

        # ------------------------------------------
        # Column presence validation
        # ------------------------------------------

        for col in expected_columns:
            if col not in df.columns:
                metrics["missing_columns"] += 1
                raise ValueError(f"Missing column: {col}")

        # ------------------------------------------
        # Primary key uniqueness
        # ------------------------------------------

        for key_tuple in zip(*(df[k] for k in primary_keys)):
            if key_tuple in seen_keys:
                metrics["duplicate_primary_keys"] += 1
            else:
                seen_keys.add(key_tuple)

        # ------------------------------------------
        # Field-level validation
        # ------------------------------------------

        all_fields: Dict[str, Any] = {}
        all_fields.update(identifiers)
        all_fields.update(features)
        all_fields.update(target)

        for field, meta in all_fields.items():

            series = df[field]

            # Nullability check
            if not meta.get("nullable", False):
                null_count = series.isnull().sum()
                metrics["null_violations"] += int(null_count)

            # Dtype validation (basic enforcement)
            expected_dtype = meta.get("dtype")

            if expected_dtype == "float32":
                if not pd.api.types.is_float_dtype(series):
                    metrics["dtype_violations"] += 1

            elif expected_dtype == "int32":
                if not pd.api.types.is_integer_dtype(series):
                    metrics["dtype_violations"] += 1

            elif expected_dtype == "string":
                if not pd.api.types.is_object_dtype(series):
                    metrics["dtype_violations"] += 1

            # Range validation
            if "min" in meta:
                metrics["range_violations"] += int((series < meta["min"]).sum())

            if "max" in meta:
                metrics["range_violations"] += int((series > meta["max"]).sum())

    # ========================================================
    # Write metrics file
    # ========================================================

    metrics_path = Path("metrics")
    metrics_path.mkdir(exist_ok=True)

    with open(metrics_path / "processed_validation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ========================================================
    # Final strict failure checks
    # ========================================================

    if metrics["duplicate_primary_keys"] > 0:
        raise ValueError("Duplicate primary keys detected")

    if metrics["null_violations"] > 0:
        raise ValueError("Null constraint violations detected")

    if metrics["range_violations"] > 0:
        raise ValueError("Range constraint violations detected")

    if metrics["dtype_violations"] > 0:
        raise ValueError("Dtype violations detected")

    print("✅ Processed dataset validation passed.")


# ============================================================

if __name__ == "__main__":
    validate()