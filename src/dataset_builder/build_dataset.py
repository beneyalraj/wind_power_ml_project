import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
import gc

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
SPEC_PATH = Path("configs/feature_spec.yaml")
INPUT_DIR = Path("data/validated_features")
OUTPUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")
METRICS_DIR = Path("metrics")

LOG_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "dataset_builder.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Load Feature Spec
# ---------------------------------------------------
def load_feature_spec():
    """Load feature specification from YAML"""
    with open(SPEC_PATH) as f:
        spec = yaml.safe_load(f)
    return spec

# ---------------------------------------------------
# Build Dataset with Validation
# ---------------------------------------------------
def build_dataset(df, spec, optimize_dtypes=True):
    """
    Build ML-ready dataset with X/y separation and schema validation
    
    Args:
        df: Input dataframe
        spec: Feature specification
        optimize_dtypes: If True, downcast to float32 (default: True)
    
    Returns:
        X: Features only (DataFrame)
        y: Target only (DataFrame)
    
    Raises:
        ValueError: If feature schema doesn't match specification
    """
    numerical = spec["numerical_features"]
    derived = spec["derived_features"]
    target = spec["target"]
    
    feature_columns = numerical + derived
    
    # CRITICAL: Schema validation
    expected_features = set(feature_columns)
    actual_features = set(df.columns) - {target}  # Exclude target from check
    
    if expected_features != actual_features:
        missing = expected_features - actual_features
        extra = actual_features - expected_features
        
        error_msg = "Feature schema mismatch!\n"
        if missing:
            error_msg += f"  Missing features: {sorted(missing)}\n"
        if extra:
            error_msg += f"  Extra features: {sorted(extra)}\n"
        error_msg += f"  Expected: {sorted(expected_features)}\n"
        error_msg += f"  Got: {sorted(actual_features)}"
        
        raise ValueError(error_msg)
    
    # Extract features (X) - preserve order from spec
    X = df[feature_columns].copy()
    
    # Extract target (y)
    y = df[[target]].copy()
    
    # Optional: Optimize data types for memory efficiency
    if optimize_dtypes:
        logger.debug("Optimizing data types to float32...")
        
        # Convert features to float32
        for col in X.columns:
            if X[col].dtype == 'float64':
                X[col] = X[col].astype('float32')
        
        # Convert target to float32
        if y[target].dtype == 'float64':
            y[target] = y[target].astype('float32')
    
    return X, y

# ---------------------------------------------------
# Process Split
# ---------------------------------------------------
def process_split(split_name, spec, optimize_dtypes=True):
    """
    Process split with X/y separation, schema validation, and memory optimization
    
    Args:
        split_name: Name of split (train/validation/test)
        spec: Feature specification
        optimize_dtypes: Whether to downcast to float32
    
    Returns:
        metrics: Dictionary with processing metrics
    """
    input_path = INPUT_DIR / split_name
    output_X_path = OUTPUT_DIR / split_name / "X"
    output_y_path = OUTPUT_DIR / split_name / "y"
    
    output_X_path.mkdir(parents=True, exist_ok=True)
    output_y_path.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(input_path.glob("*.parquet"))
    logger.info(f"Building dataset for {split_name}: {len(parquet_files)} files")
    
    # Get column list for optimized reading
    numerical = spec["numerical_features"]
    derived = spec["derived_features"]
    target = spec["target"]
    columns_to_read = numerical + derived + [target]
    
    # Metrics tracking
    metrics = {
        "split": split_name,
        "files_total": len(parquet_files),
        "files_processed": 0,
        "files_failed": 0,
        "total_rows": 0,
        "total_features": len(numerical + derived),
        "memory_saved_mb": 0,
        "timestamp": datetime.now().isoformat(),
        "optimization_enabled": optimize_dtypes,
        "errors": []
    }
    
    for i, file in enumerate(parquet_files):
        try:
            # MEMORY OPTIMIZATION: Only load needed columns (saves 30-50% memory)
            df = pd.read_parquet(file, columns=columns_to_read)
            
            # Track memory before processing
            mem_before = df.memory_usage(deep=True).sum() / 1e6
            
            # Build X and y WITH SCHEMA VALIDATION
            X, y = build_dataset(df, spec, optimize_dtypes=optimize_dtypes)
            
            # Track memory after processing
            mem_after = (X.memory_usage(deep=True).sum() + 
                        y.memory_usage(deep=True).sum()) / 1e6
            memory_saved = mem_before - mem_after
            
            # Save X (features) - separate file
            X_filename = file.stem + "_X.parquet"
            X.to_parquet(output_X_path / X_filename, index=False)
            
            # Save y (target) - separate file
            y_filename = file.stem + "_y.parquet"
            y.to_parquet(output_y_path / y_filename, index=False)
            
            # Update metrics
            metrics["files_processed"] += 1
            metrics["total_rows"] += len(X)
            metrics["memory_saved_mb"] += memory_saved
            
            logger.info(
                f"  ✓ Processed: {file.name} "
                f"({len(X):,} rows, {len(X.columns)} features, "
                f"saved {memory_saved:.2f} MB)"
            )
            
            # MEMORY MANAGEMENT: Free memory explicitly
            del df, X, y
            
            # OPTIONAL: Periodic garbage collection (every 10 files)
            # Reduces memory pressure in long-running processes
            if i % 10 == 0 and i > 0:
                collected = gc.collect()
                logger.debug(f"  Garbage collection: freed {collected} objects")
            
        except ValueError as e:
            # Schema validation error - critical, fail fast
            metrics["files_failed"] += 1
            error_details = {
                "file": file.name,
                "error_type": "schema_mismatch",
                "details": str(e)
            }
            metrics["errors"].append(error_details)
            logger.error(f"  ✗ Schema validation failed: {file.name}")
            logger.error(f"     {str(e)}")
            raise  # Fail fast on schema errors
            
        except Exception as e:
            # Other processing errors
            metrics["files_failed"] += 1
            error_details = {
                "file": file.name,
                "error_type": "processing_error",
                "details": str(e)
            }
            metrics["errors"].append(error_details)
            logger.error(f"  ✗ Failed processing {file.name}: {e}")
            raise  # Fail on any error
    
    # Final garbage collection after processing split
    gc.collect()
    
    # Save metrics for this split
    metrics_file = METRICS_DIR / f"dataset_builder_{split_name}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Summary for this split
    logger.info(f"\n✓ {split_name.upper()} split complete:")
    logger.info(f"  Files processed: {metrics['files_processed']}/{metrics['files_total']}")
    logger.info(f"  Files failed: {metrics['files_failed']}")
    logger.info(f"  Total rows: {metrics['total_rows']:,}")
    logger.info(f"  Total features: {metrics['total_features']}")
    logger.info(f"  Memory saved: {metrics['memory_saved_mb']:.2f} MB")
    
    if metrics['files_failed'] > 0:
        logger.error(f"  ⚠️  {metrics['files_failed']} files failed processing")
    
    return metrics

# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------
def main():
    logger.info("="*70)
    logger.info("DATASET BUILDER - FINAL PROCESSING STAGE")
    logger.info("="*70)
    
    # Load feature specification
    logger.info("\nLoading feature specification...")
    try:
        spec = load_feature_spec()
        logger.info(f"  ✓ Spec loaded: {SPEC_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load feature spec: {e}")
        sys.exit(1)
    
    # Log feature summary
    logger.info(f"  Numerical features: {len(spec['numerical_features'])}")
    logger.info(f"  Derived features: {len(spec['derived_features'])}")
    logger.info(f"  Target: {spec['target']}")
    
    # Determine optimization settings
    # For showcase/production: Enable optimization (float32)
    # For scientific/research: Disable optimization (float64)
    optimize_dtypes = True
    
    if optimize_dtypes:
        logger.info("  Data type optimization: ENABLED (float32)")
        logger.info("  Expected memory savings: 40-50%")
    else:
        logger.info("  Data type optimization: DISABLED (float64)")
    
    # Process each split
    logger.info("\nProcessing splits...")
    all_metrics = {}
    
    splits_to_process = ["train", "validation", "test"]
    
    for split in splits_to_process:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {split.upper()} split")
        logger.info(f"{'='*70}")
        
        try:
            metrics = process_split(split, spec, optimize_dtypes=optimize_dtypes)
            all_metrics[split] = metrics
            
        except Exception as e:
            logger.critical(f"Failed to process {split} split: {e}")
            sys.exit(1)
    
    # Final summary across all splits
    logger.info(f"\n{'='*70}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    
    total_files = sum(m['files_total'] for m in all_metrics.values())
    total_processed = sum(m['files_processed'] for m in all_metrics.values())
    total_failed = sum(m['files_failed'] for m in all_metrics.values())
    total_rows = sum(m['total_rows'] for m in all_metrics.values())
    total_memory_saved = sum(m['memory_saved_mb'] for m in all_metrics.values())
    
    logger.info(f"Total files: {total_files}")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Total memory saved: {total_memory_saved:.2f} MB")
    
    # Save combined metrics
    combined_metrics = {
        "timestamp": datetime.now().isoformat(),
        "optimization_enabled": optimize_dtypes,
        "spec_path": str(SPEC_PATH),
        "summary": {
            "total_files": total_files,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_rows": total_rows,
            "total_memory_saved_mb": round(total_memory_saved, 2)
        },
        "splits": all_metrics
    }
    
    summary_metrics_file = METRICS_DIR / "dataset_builder_summary.json"
    with open(summary_metrics_file, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    logger.info(f"\n✓ Summary metrics saved: {summary_metrics_file}")
    
    # Output structure documentation
    logger.info(f"\n{'='*70}")
    logger.info("OUTPUT STRUCTURE")
    logger.info(f"{'='*70}")
    logger.info("\nGenerated directory structure:")
    logger.info("  data/processed/")
    logger.info("    ├── train/")
    logger.info("    │   ├── X/  (feature files)")
    logger.info("    │   │   ├── layout_000_X.parquet")
    logger.info("    │   │   ├── layout_001_X.parquet")
    logger.info("    │   │   └── ...")
    logger.info("    │   └── y/  (target files)")
    logger.info("    │       ├── layout_000_y.parquet")
    logger.info("    │       ├── layout_001_y.parquet")
    logger.info("    │       └── ...")
    logger.info("    ├── validation/")
    logger.info("    │   ├── X/")
    logger.info("    │   └── y/")
    logger.info("    └── test/")
    logger.info("        ├── X/")
    logger.info("        └── y/")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ DATASET BUILDER COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*70}")
    
    # Usage instructions
    logger.info("\nNext steps:")
    logger.info("  1. Train model using: data/processed/train/X/ and data/processed/train/y/")
    logger.info("  2. Validate using: data/processed/validation/X/ and data/processed/validation/y/")
    logger.info("  3. Test using: data/processed/test/X/ and data/processed/test/y/")
    
    logger.info("\nExample loading code:")
    logger.info("  import pandas as pd")
    logger.info("  from pathlib import Path")
    logger.info("  ")
    logger.info("  # Load all training data")
    logger.info("  X_files = Path('data/processed/train/X').glob('*.parquet')")
    logger.info("  X_train = pd.concat([pd.read_parquet(f) for f in X_files])")
    logger.info("  ")
    logger.info("  y_files = Path('data/processed/train/y').glob('*.parquet')")
    logger.info("  y_train = pd.concat([pd.read_parquet(f) for f in y_files])")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)



