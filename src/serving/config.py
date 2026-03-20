"""
src/serving/config.py — Centralised configuration via pydantic-settings.

WHY THIS EXISTS:
  Hardcoded paths like Path("models/model_latest.joblib") scattered across
  app.py are a maintenance hazard. If the path changes, you grep and pray.
  More critically, hardcoded values cannot be overridden per environment —
  dev, staging, and production all need different model paths.

HOW IT WORKS:
  pydantic-settings reads values in this priority order (highest wins):
    1. Real environment variables  (production / CI)
    2. Values in .env file         (local development)
    3. Default values below        (fallback)

  This means the exact same app.py code runs in all environments — only
  the .env file or the environment variables change.

TWO SERVING MODES:
  Local dev  (use_mlflow_registry=false):
    Loads model from local joblib file. No internet required.
    Default — anyone can clone and run immediately.

  Production (use_mlflow_registry=true):
    Loads model directly from DagHub MLflow registry.
    Requires MLFLOW_TRACKING_URI, USERNAME, PASSWORD in environment.
    When a new model is promoted to "Production" in the registry,
    the next API restart picks it up automatically — no file copying.

USAGE IN app.py:
  from src.serving.config import settings

  MODEL_PATH = settings.model_path
  LOG_LEVEL  = settings.log_level
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Model Asset Paths (local dev) ---
    model_path:    Path = Path("models/model_latest.joblib")
    feature_path:  Path = Path("models/features.json")
    metadata_path: Path = Path("models/metadata.json")

    # --- API Settings ---
    api_title:   str = "Wind Power Prediction API"
    api_version: str = "1.0.0"
    log_level:   str = "INFO"

    # --- MLflow Registry (production) ---
    # Set use_mlflow_registry=true to load from DagHub instead of local file.
    # All three MLflow credentials must be set when registry mode is enabled.
    use_mlflow_registry:      bool = False
    mlflow_tracking_uri:      str  = ""
    mlflow_tracking_username: str  = ""
    mlflow_tracking_password: str  = ""
    mlflow_model_name:        str  = "WindPowerPredictor"
    mlflow_model_alias:       str = "champion"

    model_config = SettingsConfigDict(
        env_file=".env",           # load from .env if present
        env_file_encoding="utf-8",
        case_sensitive=False,      # MODEL_PATH and model_path are the same
        extra="ignore",            # silently ignore unknown env vars
    )


# Single shared instance — import this everywhere.
# Never instantiate Settings() directly in other modules.
settings = Settings()