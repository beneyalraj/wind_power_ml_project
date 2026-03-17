"""
Model Training Pipeline - Production Ready with DagsHub MLflow Tracking
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os


# DagsHub and MLflow
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    mean_absolute_percentage_error
)

# ---------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------
PARAMS_PATH = Path("params.yaml")
SPEC_PATH = Path("configs/feature_spec.yaml")
DATA_DIR = Path("data/processed")

TRAIN_X = DATA_DIR / "train" / "X"
TRAIN_Y = DATA_DIR / "train" / "y"
VAL_X = DATA_DIR / "validation" / "X"
VAL_Y = DATA_DIR / "validation" / "y"
TEST_X = DATA_DIR / "test" / "X"
TEST_Y = DATA_DIR / "test" / "y"

MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
LOG_DIR = Path("logs")
METRICS_DIR = Path("metrics")

for d in [MODEL_DIR, REPORTS_DIR, LOG_DIR, METRICS_DIR]:
    d.mkdir(exist_ok=True)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# DagsHub Configuration
# ---------------------------------------------------
def setup_dagshub():
    """
    Setup DagsHub MLflow tracking
    
    Set environment variables:
    - DAGSHUB_USER_NAME: Your DagsHub username
    - DAGSHUB_REPO_NAME: Your repository name (default: wind_power_system)
    - DAGSHUB_TOKEN: Your DagsHub access token (optional, will prompt if missing)
    """
    
    # Get DagsHub credentials from environment or use defaults
    dagshub_user = os.getenv('DAGSHUB_USER_NAME')
    dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'wind_power_system')
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if not dagshub_user:
        logger.warning(
            "DAGSHUB_USER_NAME not set. Using local MLflow tracking.\n"
            "To use DagsHub, set environment variable:\n"
            "  export DAGSHUB_USER_NAME='your_username'"
        )
        return False
    
    logger.info("="*70)
    logger.info("DAGSHUB MLFLOW SETUP")
    logger.info("="*70)
    logger.info(f"DagsHub User: {dagshub_user}")
    logger.info(f"DagsHub Repo: {dagshub_repo}")
    
    try:
        # Initialize DagsHub
        dagshub.init(
            repo_owner=dagshub_user,
            repo_name=dagshub_repo,
            mlflow=True
        )
        
        # Set MLflow tracking URI
        tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        logger.info(f"✓ MLflow tracking URI: {tracking_uri}")
        logger.info(f"✓ DagsHub initialized successfully")
        logger.info(f"  View experiments at: https://dagshub.com/{dagshub_user}/{dagshub_repo}/experiments")
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to initialize DagsHub: {e}")
        logger.warning("Falling back to local MLflow tracking")
        return False

# ---------------------------------------------------
# [Keep all your helper functions from before]
# load_config, load_feature_spec, load_dataset_safe, etc.
# ---------------------------------------------------

def load_config():
    """Load training configuration from params.yaml"""
    with open(PARAMS_PATH) as f:
        config = yaml.safe_load(f)
    return config

def load_feature_spec():
    """Load feature specification"""
    with open(SPEC_PATH) as f:
        return yaml.safe_load(f)

def load_dataset_safe(X_path, y_path, max_memory_mb=5000, load_all=True):
    """
    Safely load dataset with memory monitoring
    
    Args:
        X_path: Path to feature files
        y_path: Path to target files  
        max_memory_mb: Memory limit (warning only)
        load_all: If True, load all files regardless of memory limit
    
    Returns:
        X, y: Feature and target arrays
    """
    X_files = sorted(list(X_path.glob("*.parquet")))
    y_files = sorted(list(y_path.glob("*.parquet")))
    
    if not X_files:
        raise FileNotFoundError(f"No feature files found in {X_path}")
    if not y_files:
        raise FileNotFoundError(f"No target files found in {y_path}")
    if len(X_files) != len(y_files):
        raise ValueError(f"File count mismatch: {len(X_files)} X files vs {len(y_files)} y files")
    
    logger.info(f"Loading {len(X_files)} file(s) from {X_path.name}")
    
    X_chunks = []
    y_chunks = []
    
    for i, (X_file, y_file) in enumerate(zip(X_files, y_files)):
        # Verify file alignment
        X_base = X_file.stem.replace('_X', '')
        y_base = y_file.stem.replace('_y', '')
        
        if X_base != y_base:
            raise ValueError(
                f"File mismatch at index {i}: "
                f"{X_file.name} vs {y_file.name}"
            )
        
        # Load files
        X_chunk = pd.read_parquet(X_file)
        y_chunk = pd.read_parquet(y_file)
        
        # Validate shapes
        if len(X_chunk) != len(y_chunk):
            raise ValueError(
                f"Row count mismatch in {X_file.name}: "
                f"X={len(X_chunk)}, y={len(y_chunk)}"
            )
        
        # Check for NaNs
        if X_chunk.isnull().any().any():
            raise ValueError(f"NaN values found in {X_file.name}")
        if y_chunk.isnull().any().any():
            raise ValueError(f"NaN values found in {y_file.name}")
        
        # Append chunks
        X_chunks.append(X_chunk)
        y_chunks.append(y_chunk)
        
        # Calculate current memory usage
        current_rows = sum(len(c) for c in X_chunks)
        current_memory = sum(c.memory_usage(deep=True).sum() for c in X_chunks) / 1e6
        
        # Log progress
        logger.info(
            f"  [{i+1:>2}/{len(X_files)}] {X_file.name:30s} "
            f"| Batch: {len(X_chunk):>5,} rows "
            f"| Total: {current_rows:>6,} rows, {current_memory:>6.1f} MB"
        )
        
        # Memory warning (but don't stop if load_all=True)
        if current_memory > max_memory_mb:
            if load_all:
                logger.warning(
                    f"⚠️  Memory ({current_memory:.0f} MB) > limit ({max_memory_mb} MB) "
                    f"but continuing (load_all=True)"
                )
            else:
                logger.warning(
                    f"Memory limit reached. Stopping at {len(X_chunks)} files."
                )
                break
    
    # Concatenate all chunks
    logger.info(f"Concatenating {len(X_chunks)} chunk(s)...")
    X = pd.concat(X_chunks, ignore_index=True)
    y = pd.concat(y_chunks, ignore_index=True).values.ravel()
    
    # Final summary
    final_memory_X = X.memory_usage(deep=True).sum() / 1e6
    final_memory_y = y.nbytes / 1e6
    total_memory = final_memory_X + final_memory_y
    
    logger.info(f"✓ Dataset loaded:")
    logger.info(f"  Files:    {len(X_chunks)}/{len(X_files)}")
    logger.info(f"  Rows:     {len(X):,}")
    logger.info(f"  Features: {len(X.columns)}")
    logger.info(f"  Memory:   {total_memory:.2f} MB (X: {final_memory_X:.2f}, y: {final_memory_y:.2f})")
    
    # Final validation
    assert len(X) == len(y), f"Length mismatch: X={len(X)}, y={len(y)}"
    
    return X, y

def validate_predictions(y_true, y_pred, split_name="validation"):
    """Validate predictions for issues"""
    issues = []
    
    nan_count = np.isnan(y_pred).sum()
    if nan_count > 0:
        issues.append(f"NaN predictions: {nan_count}")
    
    neg_count = (y_pred < 0).sum()
    if neg_count > 0:
        issues.append(f"Negative predictions: {neg_count}")
    
    residuals = y_true - y_pred
    extreme_errors = np.abs(residuals) > 3 * np.std(residuals)
    if extreme_errors.sum() > len(y_true) * 0.01:
        issues.append(
            f"Extreme errors: {extreme_errors.sum()} "
            f"({extreme_errors.sum()/len(y_true)*100:.1f}%)"
        )
    
    if issues:
        logger.warning(f"Prediction issues in {split_name}: {', '.join(issues)}")
    else:
        logger.info(f"✓ Predictions validated for {split_name}")
    
    return issues

def compute_metrics(y_true, y_pred, prefix=""):
    """Compute comprehensive regression metrics"""

    y_pred_clipped = np.maximum(0, y_pred)
    metrics = {
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        f"{prefix}r2": float(r2_score(y_true, y_pred_clipped)),
    }
    
    # Add residual statistics
    residuals = y_true - y_pred_clipped
    metrics[f"{prefix}mean_residual"] = float(np.mean(residuals))
    metrics[f"{prefix}std_residual"] = float(np.std(residuals))
    
    # SMAPE instead of MAPE (handles zeros better)
    smape = np.mean(
        2 * np.abs(y_true - y_pred_clipped) / (np.abs(y_true) + np.abs(y_pred_clipped) + 1e-8)
    ) * 100
    metrics[f"{prefix}smape"] = float(smape)
    
    return metrics

def plot_predictions(y_true, y_pred, split_name, save_path):
    """Generate prediction analysis plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Power (kW)')
    axes[0].set_ylabel('Predicted Power (kW)')
    axes[0].set_title(f'{split_name} - Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residual Distribution
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(alpha=0.3)
    
    # Residual vs Predicted
    axes[2].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Predicted Power (kW)')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Residual vs Predicted')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved prediction plots: {save_path}")

def plot_feature_importance(model, feature_names, save_path):
    """Generate feature importance plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances", fontsize=16, fontweight='bold')
    sns.barplot(
        x=importances[indices], 
        y=np.array(feature_names)[indices], 
        palette="viridis"
    )
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved feature importance: {save_path}")
    
    top_features = [
        (feature_names[i], importances[i]) 
        for i in indices[:5]
    ]
    return top_features

# ---------------------------------------------------
# Training Pipeline (Updated)
# ---------------------------------------------------
def train():
    logger.info("="*70)
    logger.info("MODEL TRAINING PIPELINE - MULTI MODEL")
    logger.info("="*70)
    
    dagshub_enabled = setup_dagshub()

    config = load_config()
    spec = load_feature_spec()

    train_config = config.get('training', {})
    rf_params = train_config.get('random_forest', {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1
    })

    mlflow.set_experiment("Wind_Power_Prediction")

    # ----------------------------
    # Load datasets
    # ----------------------------
    X_train, y_train = load_dataset_safe(TRAIN_X, TRAIN_Y)
    X_val, y_val = load_dataset_safe(VAL_X, VAL_Y)
    X_test, y_test = load_dataset_safe(TEST_X, TEST_Y)

    feature_names = X_train.columns.tolist()

    # ----------------------------
    # Define models
    # ----------------------------
    from sklearn.ensemble import GradientBoostingRegressor

    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(**rf_params),
        "gboost": GradientBoostingRegressor(random_state=42)
    }

    best_model = None
    best_score = float("-inf")
    best_model_name = None
    best_metrics = {}

    # ----------------------------
    # Train all models
    # ----------------------------
    for model_name, model in models.items():

        logger.info(f"\n{'='*60}")
        logger.info(f"Training Model: {model_name}")
        logger.info(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):

            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("dagshub_enabled", str(dagshub_enabled))

            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("train_samples", len(X_train))

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            all_metrics = {}

            for split_name, X, y in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test)
            ]:

                preds = model.predict(X)

                validate_predictions(y, preds, split_name)

                metrics = compute_metrics(y, preds, prefix=f"{split_name}_")
                all_metrics.update(metrics)

                mlflow.log_metrics(metrics)

            # Overfitting check
            gap = all_metrics["train_r2"] - all_metrics["val_r2"]
            mlflow.log_metric("overfit_gap", gap)

            if gap > 0.1:
                logger.warning(f"Overfitting detected in {model_name}: gap={gap:.3f}")

            val_score = all_metrics["val_r2"]

            logger.info(f"{model_name} Validation R²: {val_score:.4f}")

            # Save feature list
            mlflow.log_dict({"features": feature_names}, "features.json")

            # Track best model
            if val_score > best_score:
                best_score = val_score
                best_model = model
                best_model_name = model_name
                best_metrics = all_metrics

    # ----------------------------
    # Log BEST model
    # ----------------------------
    logger.info(f"\n{'='*70}")
    logger.info(f"BEST MODEL: {best_model_name} (R²={best_score:.4f})")
    logger.info(f"{'='*70}")

    # ENTERPRISE THRESHOLDS (Dual-Gate)
    MIN_R2_THRESHOLD = 0.85
    MAX_SMAPE_THRESHOLD = 20.0  # Max 20% error allowed for production

    best_val_smape = best_metrics["val_smape"]

    with mlflow.start_run(run_name="best_model") as active_run:
        
        run_id = active_run.info.run_id
        
        test_preds = best_model.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds, prefix="test_")
        mlflow.log_metrics(test_metrics)

        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3)
        )

        mlflow.log_metric("best_val_r2", best_score)
        mlflow.log_metric("best_val_smape", best_val_smape)
        mlflow.log_param("best_model", best_model_name)
        
        # ---------------------------------------------------
        # THE ENTERPRISE VALIDATION GATE
        # ---------------------------------------------------
        passed_r2 = best_score >= MIN_R2_THRESHOLD
        passed_smape = best_val_smape <= MAX_SMAPE_THRESHOLD

        if passed_r2 and passed_smape:
            logger.info(f"✅ Model PASSED enterprise gate (R²: {best_score:.4f}, SMAPE: {best_val_smape:.2f}%). Registering...")
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="WindPowerPredictor"
            )
        else:
            logger.warning(f"🚨 Model FAILED enterprise gate.")
            logger.warning(f"   - Target: R² >= {MIN_R2_THRESHOLD}, SMAPE <= {MAX_SMAPE_THRESHOLD}%")
            logger.warning(f"   - Actual: R² == {best_score:.4f}, SMAPE == {best_val_smape:.2f}%")
            logger.warning("Registration aborted. Production model remains unchanged.")

    # ----------------------------
    # Save model locally
    # ----------------------------
    version = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_path = MODEL_DIR / f"model_{best_model_name}_{version}.joblib"
    joblib.dump(best_model, model_path)

    latest_path = MODEL_DIR / "model_latest.joblib"
    joblib.dump(best_model, latest_path)

    # Save feature names
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump(feature_names, f)

    # Save metrics
    metrics_path = METRICS_DIR / "training_metrics_latest.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

        logger.info(f"\nSaved best model: {model_path}")
        logger.info(f"Best model type: {best_model_name}")
        # Log run info

        # run_id = run.info.run_id
        logger.info(f"\n✓ MLflow Run ID: {run_id}")
        
        if dagshub_enabled:
            dagshub_user = os.getenv('DAGSHUB_USER_NAME')
            dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'wind_power_system')
            logger.info(
                f"✓ View this run on DagsHub:\n"
                f"  https://dagshub.com/{dagshub_user}/{dagshub_repo}/experiments"
            )
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {best_model_name}")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation R²: {best_metrics['val_r2']:.4f}")
    logger.info(f"Test R²: {test_metrics['test_r2']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['test_rmse']:,.2f} kW")
    logger.info(f"Test SMAPE: {test_metrics['test_smape']:.2f}%")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)