import os
import uuid
import logging
import json
from src.serving.config import settings
from contextlib import asynccontextmanager
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.serving.predictor import build_features

# -----------------------------
# Setup & Constants
# -----------------------------
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

MODEL_PATH    = settings.model_path
FEATURE_PATH  = settings.feature_path
METADATA_PATH = settings.metadata_path

# -----------------------------
# App State (avoids fragile globals)
# -----------------------------
class AppState:
    model         = None
    feature_names = None
    model_version = "unknown"

state = AppState()

# -----------------------------
# 1. Lifespan (replaces deprecated @app.on_event)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and graceful shutdown.

    TWO LOADING PATHS — controlled by USE_MLFLOW_REGISTRY in .env:

      false (default): loads model from local joblib file.
                       Works offline, no credentials needed.
                       Used for local dev and CI test runs.

      true:            loads model from DagHub MLflow registry.
                       Requires MLFLOW_TRACKING_URI, USERNAME, PASSWORD.
                       Used in production — when a new model version is
                       promoted to "Production" in the registry, the next
                       restart picks it up automatically.
    """
    # --- Startup ---
    try:
        if settings.use_mlflow_registry:
            # ---------------------------------------------------------
            # PRODUCTION PATH — load from DagHub MLflow registry
            # ---------------------------------------------------------
            import mlflow
            import mlflow.sklearn

            # Inject credentials so MLflow can authenticate with DagHub
            os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
            logger.info(f"Loading model from registry: {model_uri}")

            state.model = mlflow.sklearn.load_model(model_uri)
            state.model_version = f"{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
            logger.info(f"✅ Model loaded from registry: {state.model_version}")

        else:
            # ---------------------------------------------------------
            # LOCAL PATH — load from joblib file
            # ---------------------------------------------------------
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model missing at {MODEL_PATH}")
            state.model = joblib.load(MODEL_PATH)

            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    state.model_version = json.load(f).get("model_version", "unknown")

            logger.info(f"✅ Model v{state.model_version} loaded from local file.")

        # Feature names always loaded from local file in both paths
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(f"Features missing at {FEATURE_PATH}")
        with open(FEATURE_PATH, "r") as f:
            state.feature_names = json.load(f)

        logger.info(f"✅ Feature map loaded: {len(state.feature_names)} features.")

    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
        raise RuntimeError("Failed to load ML assets. Check logs.")

    yield  # App runs here

    # --- Shutdown ---
    logger.info("🛑 Shutting down. Releasing model assets.")
    state.model         = None
    state.feature_names = None


# Initialize FastAPI App
app = FastAPI(
    title=settings.api_title,
    description="Enterprise API for predicting wind farm power output.",
    version=settings.api_version,
    lifespan=lifespan,
    root_path="/api"
)

# -----------------------------
# 2. Request ID Middleware
# -----------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attaches a unique X-Request-ID to every request for end-to-end tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# -----------------------------
# 3. Strict Data Contracts (Request + Response)
# -----------------------------
class PredictionRequest(BaseModel):
    wind_speed:           float = Field(..., ge=0.0,           description="Wind speed in m/s (Cannot be negative)")
    wind_direction:       float = Field(..., ge=0.0, le=360.0, description="Wind direction in degrees (0-360)")
    turbulence_intensity: float = Field(..., ge=0.0, le=1.0,   description="Turbulence intensity (0.0 to 1.0)")
    num_turbines:         int   = Field(..., gt=0,              description="Number of turbines must be at least 1")

    model_config = {
        "json_schema_extra": {
            "example": {
                "wind_speed": 12.5,
                "wind_direction": 180.0,
                "turbulence_intensity": 0.08,
                "num_turbines": 50
            }
        }
    }

class PredictionResponse(BaseModel):
    status:        str
    prediction_kw: float
    model_version: str
    request_id:    str

class BatchPredictionResponse(BaseModel):
    status:         str
    predictions_kw: List[float]
    count:          int
    model_version:  str
    request_id:     str

# -----------------------------
# Shared prediction logic
# -----------------------------
def _run_prediction(records: List[dict], request_id: str) -> List[float]:
    """
    Core prediction logic shared by /predict and /predict_batch.
    Accepts a list of raw input dicts, returns a list of clipped predictions.
    """
    features = build_features(records, state.feature_names)
    raw_predictions = state.model.predict(features)
    return [round(max(0.0, float(p)), 2) for p in raw_predictions]

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health_check():
    """Load balancers use this to verify the API is alive and the model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": state.model is not None,
        "model_version": state.model_version
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_power(data: PredictionRequest, request: Request):
    """Single-record prediction endpoint."""
    request_id = getattr(request.state, "request_id", "unknown")

    if state.model is None or state.feature_names is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    logger.info(f"[{request_id}] /predict request: {data.model_dump()}")

    try:
        results = _run_prediction([data.model_dump()], request_id)
        final_prediction = results[0]

        logger.info(f"[{request_id}] /predict result: {final_prediction} kW (model v{state.model_version})")

        return PredictionResponse(
            status="success",
            prediction_kw=final_prediction,
            model_version=state.model_version,
            request_id=request_id
        )

    except ValueError as ve:
        logger.error(f"[{request_id}] Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_power_batch(data: List[PredictionRequest], request: Request):
    """
    Batch prediction endpoint — accepts up to 500 records per call.
    Passes the entire batch to model.predict() in one vectorized call,
    which is significantly faster than calling /predict N times.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if state.model is None or state.feature_names is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least one record.")

    if len(data) > 500:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 500 records.")

    logger.info(f"[{request_id}] /predict_batch request: {len(data)} records.")

    try:
        records = [item.model_dump() for item in data]
        predictions = _run_prediction(records, request_id)

        logger.info(f"[{request_id}] /predict_batch result: {len(predictions)} predictions (model v{state.model_version})")

        return BatchPredictionResponse(
            status="success",
            predictions_kw=predictions,
            count=len(predictions),
            model_version=state.model_version,
            request_id=request_id
        )

    except ValueError as ve:
        logger.error(f"[{request_id}] Batch validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[{request_id}] Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction.")