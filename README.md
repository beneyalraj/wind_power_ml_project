# 🌬️ Wind Power Prediction System

A production-grade ML system for predicting wind farm power output. Built with a full MLOps pipeline — from raw data ingestion through model training, registry, and live API serving with a Streamlit dashboard.

[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen)](https://github.com/beneyalraj/wind_power_ml_project)
[![Model](https://img.shields.io/badge/model-GradientBoosting-blue)](https://dagshub.com/beneyalraj/wind_power_system)
[![Registry](https://img.shields.io/badge/registry-DagHub%20MLflow-orange)](https://dagshub.com/beneyalraj/wind_power_system)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://github.com/beneyalraj/wind_power_ml_project)

---

## What This System Does

Given wind conditions at a farm — speed, direction, turbulence, and turbine count — the system predicts total power output in kilowatts. A GradientBoosting model (R² = 0.97 on test) is served via a FastAPI backend with a Streamlit dashboard for interactive exploration.

The model is trained through a fully automated DVC pipeline triggered by GitHub Actions, validated against enterprise quality gates, and registered to a DagHub MLflow registry. The serving layer loads the model directly from the registry at startup — no manual file copying between environments.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
│                                                                 │
│   AWS S3 (raw data)                                             │
│       │                                                         │
│       ▼                                                         │
│   DVC Pipeline (GitHub Actions triggers on push)                │
│       │                                                         │
│       ├── data_ingestion        load HDF5 from S3               │
│       ├── validate_raw          schema + quality checks         │
│       ├── extract_scenario      HDF5 → partitioned Parquet      │
│       ├── validate_processed    processed data checks           │
│       ├── dataset_split         train / validation / test       │
│       ├── feature_extraction    physics-informed features       │
│       ├── feature_validation    feature schema validation       │
│       ├── build_dataset         X/y separation, W → kW          │
│       └── model_training        4 models → best selected        │
│               │                                                 │
│               ├── Enterprise Gate: R² ≥ 0.85, SMAPE ≤ 20%      │
│               │       PASS → register to MLflow                 │
│               │       FAIL → abort, production unchanged        │
│               └── DVC push → DagHub remote storage             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DAGSHUB MLFLOW REGISTRY                   │
│                                                                 │
│   Model: WindPowerPredictor                                     │
│   Alias: @champion  (production-ready version)                  │
│   URL:   https://dagshub.com/beneyalraj/wind_power_system       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER                            │
│                                                                 │
│   Docker Compose                                                │
│   ┌─────────────────────┐   ┌─────────────────────────────┐    │
│   │  FastAPI  :8000     │   │  Streamlit  :8501           │    │
│   │                     │   │                             │    │
│   │  POST /predict      │◄──│  Input sliders              │    │
│   │  POST /predict_batch│   │  Power curve chart          │    │
│   │  GET  /health       │   │  System information         │    │
│   │                     │   │                             │    │
│   │  • Pydantic v2      │   │  API_URL=http://fastapi:8000│    │
│   │  • Request tracing  │   └─────────────────────────────┘    │
│   │  • Model versioning │                                       │
│   │  • Domain clipping  │                                       │
│   └─────────────────────┘                                       │
│            │                                                    │
│            ▼                                                    │
│   Evidently AI — prediction + data drift monitoring            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                        AWS EC2 t2.micro
                        (Docker host — free tier)
```

---

## Model Performance

| Metric | Value | Gate |
|---|---|---|
| Algorithm | GradientBoosting | — |
| Features | 9 (4 raw + 5 engineered) | — |
| Training samples | 3,500 | — |
| Validation R² | 0.9282 | ✅ ≥ 0.85 |
| Test R² | 0.9714 | ✅ |
| Test RMSE | 43,870 kW | — |
| Validation SMAPE | 15.97% | ✅ ≤ 20% |

Models are only registered to production if they pass **both** gate thresholds. If a model fails, the registry is unchanged and the previous version continues serving.

---

## Project Structure

```
wind_power_system/
│
├── src/
│   ├── core/
│   │   └── config_manager.py          # centralised config loading
│   │
│   ├── ingestion/
│   │   ├── load_data.py               # downloads HDF5 from AWS S3
│   │   └── inspect_h5.py              # HDF5 structure inspection utility
│   │
│   ├── data/
│   │   ├── extract/
│   │   │   ├── extract_to_parquet.py  # HDF5 → partitioned Parquet
│   │   │   ├── h5_reader.py           # HDF5 reader utility
│   │   │   └── partitioned_writer.py  # partitioned Parquet writer
│   │   └── validate/
│   │       ├── validate_raw.py        # raw data schema + quality checks
│   │       └── validate_processed.py  # processed data validation
│   │
│   ├── dataset/
│   │   └── split_dataset.py           # train / validation / test split
│   │
│   ├── dataset_builder/
│   │   └── build_dataset.py           # X/y separation, W → kW conversion
│   │
│   ├── features/
│   │   └── build_features.py          # physics-informed feature engineering
│   │
│   ├── validation/
│   │   └── validate_features.py       # feature schema validation
│   │
│   ├── training/
│   │   └── train_model.py             # multi-model training, gate check,
│   │                                  # MLflow registration
│   └── serving/
│       ├── app.py                     # FastAPI — endpoints, middleware,
│       │                              # lifespan, dual registry/local loading
│       ├── config.py                  # pydantic-settings, .env support
│       ├── predictor.py               # feature engineering for inference
│       └── streamlit_app.py           # interactive dashboard
│
├── tests/
│   ├── conftest.py                    # shared fixtures, mock model/registry
│   ├── test_api.py                    # 27 API integration tests
│   └── test_predictor.py              # 11 predictor unit tests
│
├── configs/
│   └── feature_spec.yaml              # feature schema and derived features
│
├── models/
│   ├── model_latest.joblib            # latest trained model (local fallback)
│   ├── features.json                  # ordered feature names
│   └── metadata.json                  # model version metadata
│
├── dvc.yaml                           # 9-stage DVC pipeline definition
├── dvc.lock                           # pipeline state lock
├── params.yaml                        # training hyperparameters
├── pytest.ini                         # test discovery config
├── requirements.txt                   # pinned production dependencies
├── requirements-dev.txt               # pinned dev/test dependencies
├── .env.example                       # config template (copy to .env)
└── README.md
```

---

## Engineered Features

The model uses 4 raw inputs plus 5 physics-informed derived features:

| Feature | Type | Description |
|---|---|---|
| `wind_speed` | Raw | Wind speed at hub height (m/s) |
| `wind_direction` | Raw | Meteorological direction (°) |
| `turbulence_intensity` | Raw | σ(wind) / mean(wind) |
| `num_turbines` | Raw | Active turbines in farm |
| `wind_direction_sin` | Derived | sin(direction) — cyclical encoding |
| `wind_direction_cos` | Derived | cos(direction) — cyclical encoding |
| `wind_speed_squared` | Derived | v² — aerodynamic drag proxy |
| `wind_speed_cubed` | Derived | v³ — power law (P ∝ v³) |
| `wake_adjusted_wind` | Derived | v × (1 − TI) — turbulence wake proxy |

---

## Quick Start — Local Development

### Prerequisites
- Python 3.10+
- Git

### 1. Clone and install

```bash
git clone https://github.com/beneyalraj/wind_power_ml_project.git
cd wind_power_ml_project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# .env is pre-configured for local mode (USE_MLFLOW_REGISTRY=false)
# No credentials needed to run locally
```

### 3. Run the API

```bash
uvicorn src.serving.app:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Run the dashboard

```bash
# In a second terminal
streamlit run src/serving/streamlit_app.py
# Dashboard at http://localhost:8501
```

### 5. Run tests

```bash
pytest tests/ -v
# Expected: 38 passed
```

---

## Production Mode — Load Model from Registry

To load the model directly from DagHub MLflow registry instead of the local file, update `.env`:

```bash
USE_MLFLOW_REGISTRY=true
MLFLOW_TRACKING_URI=https://dagshub.com/beneyalraj/wind_power_system.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

Restart the API — it will download and load `WindPowerPredictor@champion` from the registry at startup. No code changes required between local and production modes.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — returns model loaded status and version |
| `/predict` | POST | Single record inference — returns kW with request trace ID |
| `/predict_batch` | POST | Batch inference — up to 500 records, vectorised |
| `/docs` | GET | Interactive Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "wind_speed": 12.5,
    "wind_direction": 180.0,
    "turbulence_intensity": 0.08,
    "num_turbines": 50
  }'
```

### Example Response

```json
{
  "status": "success",
  "prediction_kw": 172614.3,
  "model_version": "WindPowerPredictor@champion",
  "request_id": "e15a5021-afaf-49c1-80e9-17355fe2b147"
}
```

Every response includes a `request_id` (UUID4) attached via middleware and echoed in the `X-Request-ID` response header for end-to-end tracing.

---

## DVC Pipeline

The training pipeline is defined in `dvc.yaml` with 9 stages:

```
data_ingestion → validate_raw → extract_scenario_dataset
    → validate_processed → dataset_split → feature_extraction
    → feature_validation → build_dataset → model_training
```

Run the full pipeline:
```bash
dvc repro
```

Run from a specific stage (DVC skips unchanged upstream stages automatically):
```bash
dvc repro model_training
```

Push artifacts to DagHub remote:
```bash
dvc push
```

---

## Promoting a Model to Production

After training registers a new model version, promote it to the `@champion` alias via the DagHub UI or:

```python
import mlflow, dagshub
dagshub.init(repo_owner='beneyalraj', repo_name='wind_power_system', mlflow=True)
mlflow.MlflowClient().set_registered_model_alias(
    name='WindPowerPredictor',
    alias='champion',
    version='<version_number>'
)
```

The API picks up the new model on next restart — no code changes, no file copying.

---

## Links

- **GitHub:** https://github.com/beneyalraj/wind_power_ml_project
- **DagHub / MLflow:** https://dagshub.com/beneyalraj/wind_power_system
- **Swagger UI:** `http://localhost:8000/docs` (when running locally)
