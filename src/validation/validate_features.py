"""
Feature Validation Pipeline - Industry Grade
Memory-safe, scalable validation with comprehensive checks
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandera.pandas as pa
from pandera import Column, Check
import yaml
import json
from scipy import stats
import hashlib
from datetime import datetime
import random

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
SPEC_PATH = Path("configs/feature_spec.yaml")
INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/validated_features")
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
        logging.FileHandler(LOG_DIR / "feature_validation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Alerting System
# ---------------------------------------------------
class LogAlert:
    """Standardizes high-visibility alerts for log-based monitoring"""
    HEADER = "🚨 [DATA-ALERT]"
    
    @staticmethod
    def trigger(message, level="WARNING"):
        alert_msg = f"{LogAlert.HEADER} {message}"
        if level == "CRITICAL":
            logger.critical(f"\n{'!'*70}\n{alert_msg}\n{'!'*70}")
        else:
            logger.warning(alert_msg)

# ---------------------------------------------------
# Load Feature Spec
# ---------------------------------------------------
def load_feature_spec():
    """Load feature specification from YAML"""
    with open(SPEC_PATH) as f:
        spec = yaml.safe_load(f)
    
    # Add validation metadata
    spec['_validation_timestamp'] = datetime.now().isoformat()
    spec['_spec_hash'] = hashlib.md5(
        str(spec).encode()
    ).hexdigest()
    
    return spec

# ---------------------------------------------------
# Outlier Detection
# ---------------------------------------------------
def detect_outliers_iqr(series):
    """Return proportion of outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((series < lower) | (series > upper)).sum()
    return outliers / len(series) if len(series) > 0 else 0

# ---------------------------------------------------
# Build Schema
# ---------------------------------------------------
def build_schema(spec):
    """Build Pandera schema with comprehensive validation checks"""
    
    columns = {}
    
    # Identifiers (nullable=False handles not null)
    for col in spec["identifiers"]:
        columns[col] = Column(
            pa.String,
            nullable=False  # ← CHANGED: Use nullable instead of Check.not_null()
        )
    
    # Numerical features with domain constraints
    feature_constraints = spec.get("feature_constraints", {})
    
    for col in spec["numerical_features"]:
        checks = []
        
        # Add range checks if defined
        if col in feature_constraints:
            min_val = feature_constraints[col].get("min")
            max_val = feature_constraints[col].get("max")
            if min_val is not None and max_val is not None:
                checks.append(
                    Check.in_range(min_val, max_val, 
                                   include_min=True, 
                                   include_max=True)
                )
        
        # Variance check (prevent constant columns)
        if col != "num_turbines":
            checks.append(
                Check(lambda s: s.std() > 0.0001, 
                    error=f"{col} has zero or near-zero variance")
        )
        
        columns[col] = Column(
            pa.Float, 
            checks=checks,
            nullable=False  # ← CHANGED: Use nullable instead of Check.not_null()
        )
    
    # Derived features
    for col in spec["derived_features"]:
        columns[col] = Column(
            pa.Float,
            nullable=False  # ← CHANGED
        )
    
    # Target variable
    target = spec["target"]
    columns[target] = Column(
        pa.Float,
        checks=[
            Check.greater_than_or_equal_to(0, 
                error="Target (power) cannot be negative")
        ],
        nullable=False  # ← CHANGED
    )
    
    schema = pa.DataFrameSchema(columns, strict=True)
    return schema

# ---------------------------------------------------
# Streaming Statistics
# ---------------------------------------------------
def compute_profile_from_running_stats(stats):
    """Compute statistics from streaming aggregates"""
    n = stats['count']
    
    if n == 0:
        return {"error": "No data processed"}
    
    profile = {
        "total_rows": n,
        "features": {}
    }
    
    for col in stats['sum'].keys():
        mean = stats['sum'][col] / n
        variance = (stats['sum_sq'][col] / n) - (mean ** 2)
        std = variance ** 0.5 if variance > 0 else 0
        
        profile["features"][col] = {
            "dtype": "float64",
            "mean": float(mean),
            "std": float(std),
            "min": float(stats['min'][col]),
            "max": float(stats['max'][col]),
            "null_count": int(stats.get('null_count', {}).get(col, 0)),
            "zero_count": int(stats.get('zero_count', {}).get(col, 0)),
            "zero_pct": float(stats.get('zero_count', {}).get(col, 0) / n * 100) if n > 0 else 0
        }
    
    return profile

# ---------------------------------------------------
# Memory-Safe Validation
# ---------------------------------------------------
def validate_split(split_name, schema, spec, collect_stats=False):
    """
    Validate split without loading all data into memory.
    Processes one file at a time and uses streaming statistics.
    """
    
    input_path = INPUT_DIR / split_name
    output_path = OUTPUT_DIR / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(input_path.glob("*.parquet"))
    logger.info(f"Validating {split_name} split: {len(files)} files")
    
    validation_results = {
        "split": split_name,
        "files_total": len(files),
        "files_validated": 0,
        "files_failed": 0,
        "total_rows": 0,
        "errors": [],
        "low_variance_files": []
    }
    
    # Streaming statistics (no full concat)
    running_stats = {
        'count': 0,
        'sum': {},
        'sum_sq': {},
        'min': {},
        'max': {},
        'null_count': {},
        'zero_count': {}
    }
    
    target_col = spec['target']
    
    for file in files:
        try:
            # Load single file
            df = pd.read_parquet(file)
            
            # Cast num_turbines to float to match the Pandera schema
            if "num_turbines" in df.columns:
                df["num_turbines"] = df["num_turbines"].astype("float64")
            # -----------------------

            # Schema validation
            schema.validate(df, lazy=True)
            
            # Per-file target variance warning
            if target_col in df.columns:
                target_std = df[target_col].std()
                if target_std < 0.01:
                    warning_msg = (
                        f"Low target variance in {file.name}: "
                        f"std={target_std:.4f} "
                        f"(min={df[target_col].min():.2f}, "
                        f"max={df[target_col].max():.2f})"
                    )
                    logger.warning(f"  ⚠️  {warning_msg}")
                    validation_results["low_variance_files"].append({
                        "file": file.name,
                        "target_std": float(target_std),
                        "target_min": float(df[target_col].min()),
                        "target_max": float(df[target_col].max())
                    })
            
            # Update running statistics (streaming)
            if collect_stats:
                for col in df.select_dtypes(include=['float', 'int']).columns:
                    if col not in running_stats['sum']:
                        running_stats['sum'][col] = 0
                        running_stats['sum_sq'][col] = 0
                        running_stats['min'][col] = float('inf')
                        running_stats['max'][col] = float('-inf')
                        running_stats['null_count'][col] = 0
                        running_stats['zero_count'][col] = 0
                    
                    running_stats['sum'][col] += df[col].sum()
                    running_stats['sum_sq'][col] += (df[col] ** 2).sum()
                    running_stats['min'][col] = min(running_stats['min'][col], df[col].min())
                    running_stats['max'][col] = max(running_stats['max'][col], df[col].max())
                    running_stats['null_count'][col] += df[col].isnull().sum()
                    running_stats['zero_count'][col] += (df[col] == 0).sum()
                
                running_stats['count'] += len(df)
            
            # Copy to output
            df.to_parquet(output_path / file.name, index=False)
            
            validation_results["files_validated"] += 1
            validation_results["total_rows"] += len(df)
            logger.info(f"  ✓ Validated: {file.name} ({len(df):,} rows)")
            
            # FREE MEMORY IMMEDIATELY
            del df
            
        except pa.errors.SchemaErrors as e:
            validation_results["files_failed"] += 1
            error_details = {
                "file": file.name,
                "type": "schema_error",
                "details": str(e.failure_cases) if hasattr(e, 'failure_cases') else str(e)
            }
            validation_results["errors"].append(error_details)
            logger.error(f"  ✗ Schema validation failed: {file.name}")
            logger.error(f"     {str(e)[:200]}")
            
        except Exception as e:
            validation_results["files_failed"] += 1
            error_details = {
                "file": file.name,
                "type": "unexpected_error",
                "details": str(e)
            }
            validation_results["errors"].append(error_details)
            logger.error(f"  ✗ Unexpected error: {file.name}: {e}")
    
    # Calculate final statistics from running stats
    if collect_stats and running_stats['count'] > 0:
        profile = compute_profile_from_running_stats(running_stats)
        validation_results["profile"] = profile
        
        # Global target variance check
        if target_col in running_stats['sum']:
            n = running_stats['count']
            mean = running_stats['sum'][target_col] / n
            variance = (running_stats['sum_sq'][target_col] / n) - (mean ** 2)
            global_std = variance ** 0.5 if variance > 0 else 0
            
            validation_results["global_target_std"] = float(global_std)
            logger.info(f"  Global target std: {global_std:.4f}")
            
            if global_std < 0.01:
                LogAlert.trigger(
                    f"Low target variance in {split_name} split: "
                    f"std={global_std:.4f} - model may not learn effectively",
                    level="CRITICAL"
                )
        
        # Calculate data hash (lightweight version using stats)
        stats_str = json.dumps(profile, sort_keys=True)
        validation_results["data_hash"] = hashlib.md5(
            stats_str.encode()
        ).hexdigest()
    
    return validation_results

# ---------------------------------------------------
# Sample-Based Distribution Check
# ---------------------------------------------------
def sample_from_parquet_dir(directory, sample_size=10000):
    """Efficiently sample from partitioned Parquet without loading all"""
    
    files = list(Path(directory).glob("*.parquet"))
    
    # FIX 1: Defensive check for empty directory
    if not files:
        logger.warning(f"No parquet files found in {directory}")
        return pd.DataFrame()
    
    # Sample files first (faster than reading all)
    num_files_to_sample = min(len(files), max(3, len(files) // 10))
    
    # FIX 1: Extra safety (technically redundant but defensive)
    if num_files_to_sample == 0:
        logger.warning(f"Calculated 0 files to sample from {directory}")
        return pd.DataFrame()
    
    sampled_files = random.sample(files, num_files_to_sample)
    
    logger.info(f"  Sampling from {num_files_to_sample} files...")
    
    # Read samples from each file
    samples = []
    rows_per_file = sample_size // len(sampled_files)
    
    for file in sampled_files:
        df = pd.read_parquet(file)
        
        # Sample rows
        n_sample = min(len(df), rows_per_file)
        sample = df.sample(n=n_sample, random_state=42)
        samples.append(sample)
        
        del df  # Free immediately
    
    # Combine samples
    if not samples:
        logger.warning(f"No samples collected from {directory}")
        return pd.DataFrame()
    
    combined = pd.concat(samples, ignore_index=True)
    
    # Trim to exact sample size
    if len(combined) > sample_size:
        combined = combined.sample(n=sample_size, random_state=42)
    
    logger.info(f"  ✓ Sampled {len(combined):,} rows")
    
    return combined

def check_distribution_similarity_sampled(
    train_dir, val_dir, test_dir, 
    sample_size=10000, 
    threshold=0.01
):
    """
    Check distribution similarity using SAMPLES instead of full data
    Memory-efficient for large datasets
    """
    
    logger.info(f"\nChecking distribution similarity (sample size: {sample_size:,})...")
    
    # Sample from each split
    train_sample = sample_from_parquet_dir(train_dir, sample_size)
    val_sample = sample_from_parquet_dir(val_dir, sample_size)
    test_sample = sample_from_parquet_dir(test_dir, sample_size)
    
    if train_sample.empty or val_sample.empty or test_sample.empty:
        logger.warning("One or more splits have no data. Skipping distribution check.")
        return {}
    
    results = {}
    numeric_cols = train_sample.select_dtypes(include=['float', 'int']).columns
    
    drift_count = 0
    
    for col in numeric_cols:
        # KS test: train vs val
        ks_stat_val, p_val_val = stats.ks_2samp(
            train_sample[col].dropna(),
            val_sample[col].dropna()
        )
        
        # KS test: train vs test
        ks_stat_test, p_val_test = stats.ks_2samp(
            train_sample[col].dropna(),
            test_sample[col].dropna()
        )
        
        results[col] = {
            "train_vs_val": {
                "ks_statistic": float(ks_stat_val),
                "p_value": float(p_val_val),
                "similar": bool(p_val_val > threshold)
            },
            "train_vs_test": {
                "ks_statistic": float(ks_stat_test),
                "p_value": float(p_val_test),
                "similar": bool(p_val_val > threshold)
            }
        }
        
        # LOG-BASED ALERTING
        if p_val_val < threshold:
            drift_count += 1
            LogAlert.trigger(
                f"DRIFT DETECTED: {col} distribution differs between Train & Val "
                f"(KS={ks_stat_val:.4f}, p={p_val_val:.4f})"
            )
        
        if p_val_test < threshold:
            drift_count += 1
            LogAlert.trigger(
                f"DRIFT DETECTED: {col} distribution differs between Train & Test "
                f"(KS={ks_stat_test:.4f}, p={p_val_test:.4f})",
                level="CRITICAL"
            )
    
    # Summary
    logger.info(f"Distribution check complete: {drift_count} drift warnings")
    
    # Free memory
    del train_sample, val_sample, test_sample
    
    return results

# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------
def main():
    logger.info("="*70)
    logger.info("FEATURE VALIDATION PIPELINE (Memory-Safe)")
    logger.info("="*70)
    
    # Load spec
    logger.info("\nLoading feature specification...")
    spec = load_feature_spec()
    logger.info(f"  Spec version: {spec.get('version', 'unknown')}")
    logger.info(f"  Spec hash: {spec['_spec_hash']}")
    
    # Build schema
    logger.info("\nBuilding validation schema...")
    schema = build_schema(spec)
    logger.info(f"  Schema columns: {len(schema.columns)}")
    
    # Validate each split
    all_results = {
        "metadata": {
            "validation_timestamp": spec['_validation_timestamp'],
            "spec_hash": spec['_spec_hash'],
            "spec_path": str(SPEC_PATH)
        },
        "splits": {}
    }
    
    splits_to_validate = ["train", "validation", "test"]
    
    for split in splits_to_validate:
        logger.info(f"\n{'='*70}")
        logger.info(f"Validating {split.upper()} split")
        logger.info(f"{'='*70}")
        
        # Validate (streaming, no full concat)
        results = validate_split(split, schema, spec, collect_stats=True)
        all_results["splits"][split] = results
        
        # Check for failures
        if results['files_failed'] > 0:
            LogAlert.trigger(
                f"Validation FAILED for {split} split. "
                f"{results['files_failed']}/{results['files_total']} files failed.",
                level="CRITICAL"
            )
            
            # Log first few errors
            for error in results['errors'][:3]:
                logger.error(f"  Error in {error['file']}: {error['details'][:100]}")
            
            sys.exit(1)
        
        # Success summary
        logger.info(f"\n✓ {split.upper()} validation PASSED")
        logger.info(f"  Files validated: {results['files_validated']}/{results['files_total']}")
        logger.info(f"  Total rows: {results['total_rows']:,}")
        
        if 'profile' in results:
            logger.info(f"  Data hash: {results.get('data_hash', 'N/A')}")
        
        # Warn about low variance files
        if results.get('low_variance_files'):
            logger.warning(
                f"  ⚠️  {len(results['low_variance_files'])} files with low target variance"
            )
    
    # Distribution similarity check (using samples)
    logger.info(f"\n{'='*70}")
    logger.info("DISTRIBUTION SIMILARITY CHECK")
    logger.info(f"{'='*70}")
    
    similarity_results = check_distribution_similarity_sampled(
        INPUT_DIR / "train",
        INPUT_DIR / "validation",
        INPUT_DIR / "test",
        sample_size=10000,
        threshold=0.01
    )
    
    all_results["distribution_similarity"] = similarity_results
    
    # Save comprehensive metrics
    logger.info(f"\n{'='*70}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*70}")
    
    metrics_path = METRICS_DIR / "feature_validation.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"✓ Metrics saved: {metrics_path}")
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*70}")
    
    total_files = sum(r['files_total'] for r in all_results['splits'].values())
    total_validated = sum(r['files_validated'] for r in all_results['splits'].values())
    total_rows = sum(r['total_rows'] for r in all_results['splits'].values())
    
    logger.info(f"Total files validated: {total_validated}/{total_files}")
    logger.info(f"Total rows validated: {total_rows:,}")
    
    # Count distribution warnings
    drift_warnings = 0
    for col_results in similarity_results.values():
        if not col_results.get('train_vs_val', {}).get('similar', True):
            drift_warnings += 1
        if not col_results.get('train_vs_test', {}).get('similar', True):
            drift_warnings += 1
    
    if drift_warnings > 0:
        logger.warning(f"Distribution drift warnings: {drift_warnings}")
    else:
        logger.info("✓ No distribution drift detected")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ FEATURE VALIDATION COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)