# 🌬️ Wind Power Prediction System

A production-grade ML system for predicting wind farm power output. Built with a full MLOps pipeline — from raw data ingestion through model training, registry, and live API serving with a Streamlit dashboard.

[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen)](https://github.com/beneyalraj/wind_power_ml_project)
[![Model](https://img.shields.io/badge/model-GradientBoosting-blue)](https://dagshub.com/beneyalraj/wind_power_system)
[![Registry](https://img.shields.io/badge/registry-DagHub%20MLflow-orange)](https://dagshub.com/beneyalraj/wind_power_system)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://github.com/beneyalraj/wind_power_ml_project)
[![Docker](https://img.shields.io/badge/containers-Docker-2496ED)](https://github.com/beneyalraj/wind_power_ml_project)
[![AWS EC2](https://img.shields.io/badge/deployed-AWS%20EC2-FF9900)](http://13.219.46.116)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)](https://github.com/beneyalraj/wind_power_ml_project/actions)

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| Streamlit Dashboard | http://13.219.46.116 |
| FastAPI Swagger UI | http://13.219.46.116/api/docs |
| Health Check | http://13.219.46.116/api/health |
| Observability | http://13.219.46.116/report |


<img width="1433" height="811" alt="Screenshot 2026-03-21 at 8 55 49 PM" src="https://github.com/user-attachments/assets/6ad4dafa-8ee2-45b2-bcbd-23bc0cafb06c" />
<img width="1433" height="695" alt="Screenshot 2026-03-21 at 8 56 52 PM" src="https://github.com/user-attachments/assets/bed3077f-2550-4767-b2b7-994d2345e9f0" />

---

## Business Context

Accurate wind power forecasting is critical for modern energy grid operations. Grid operators need to balance supply and demand in real time — when wind output is unpredictable, they rely on fossil-fuel backup generators to compensate, increasing costs and carbon emissions.

This system predicts wind farm power output from atmospheric conditions, enabling operators to:
- **Pre-schedule generation** — commit accurate power volumes to energy markets hours in advance
- **Reduce backup reliance** — minimise expensive and polluting peaker plant activation
- **Optimise turbine layouts** — simulate output across different farm configurations before construction

---

## Tech Stack

| Category | Technologies |
|---|---|
| **ML & Data** | scikit-learn, pandas, numpy, scipy, Pandera, Joblib|
| **MLOps** | DVC, MLflow, DagHub, Evidently AI |
| **Serving** | FastAPI, Pydantic v2, Uvicorn, Streamlit, Loguru |
| **Infrastructure** | Docker, Nginx, AWS EC2, GitHub Actions|
| **Orchestration** | Apache Airflow, Linux Crontab |
| **Storage** | AWS S3, DagHub remote |
| **Language** | Python 3.10 |
| **Testing** | Pytest, HTTPX |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                        │
│                                                                 │
│   AWS S3 (raw data — NREL Wind AI Bench dataset)                │
│       │                                                         │
│       ▼                                                         │
│   Apache Airflow (local orchestration)                          │
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
│               ├── Enterprise Gate: R² ≥ 0.85, SMAPE ≤ 20%       │
│               │       PASS → register to MLflow                 │
│               │       FAIL → abort, production unchanged        │
│               └── DVC push → DagHub remote storage              │
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
│   Docker Compose (AWS EC2 t3.micro)                             │
│   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│   │    Nginx     │  │    FastAPI      │  │   Streamlit      │   │
│   │   :80        │  │   :8000         │  │   :8501          │   │
│   │              │  │                 │  │                  │   │
│   │ /api/* ────► │  │ POST /predict   │◄─│ Input sliders    │   │
│   │ /* ───────►  │  │ POST /predict   │  │ Power curve      │   │
│   │ /report ──┐  │  │   _batch        │  │ System info      │   │
│   │ (HTML)    │  │  │ GET  /health    │  │                  │   │
│   │           │  │  │ Async Telemetry │  │ API_URL=         │   │
│   │ Reverse   │  │  │(BackgroundTasks)│  │ http://fastapi   │   │
│   │ Proxy     │  │  └────────┬────────┘  │ :8000            │   │
│   └─────▲────────┘           │           └──────────────────┘   │
│         │                    │ (JSONL logs)                     │
│         │                    ▼                                  │
│         │      ┌───────────────────────────┐                    │
│         │      │  Shared Docker Volume     │                    │
│         │      └─────────────┬─────────────┘                    │
│         │                    │                                  │
│         │                    ▼                                  │
│         │      ┌───────────────────────────┐                    │
│         └──────┤  Evidently AI Container   │                    │
│                │  (Cron @ Weekly)          │                    │
│                └───────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```
---

## Dataset

This project uses the **NREL Wind AI Bench** dataset, a publicly available benchmark for wind energy machine learning research.

| | |
|---|---|
| **Source** | [NREL Wind AI Bench — AWS Open Data Registry](https://registry.opendata.aws/nrel-pds-windai) |
| **License** | Creative Commons Attribution 4.0 United States |
| **Managed by** | National Renewable Energy Laboratory (NREL) |
| **Contact** | Ryan King — ryan.king@nrel.gov |
| **Documentation** | https://github.com/NREL/windAI_bench |

**Citation:**
> Wind AI Bench was accessed on 2025 from https://registry.opendata.aws/nrel-pds-windai

---

## Model Performance

<here goes your mlflow experiments image>

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

## Project Structure

```
wind_power_system/
│
├── src/
│   ├── core/
│   │   └── config_manager.py          # centralised config loading
│   ├── ingestion/
│   │   ├── load_data.py               # downloads HDF5 from AWS S3
│   │   └── inspect_h5.py              # HDF5 structure inspection utility
│   ├── data/
│   │   ├── extract/
│   │   │   ├── extract_to_parquet.py  # HDF5 → partitioned Parquet
│   │   │   ├── h5_reader.py           # HDF5 reader utility
│   │   │   └── partitioned_writer.py  # partitioned Parquet writer
│   │   └── validate/
│   │       ├── validate_raw.py        # raw data schema + quality checks
│   │       └── validate_processed.py  # processed data validation
│   ├── dataset/
│   │   └── split_dataset.py           # train / validation / test split
│   ├── dataset_builder/
│   │   └── build_dataset.py           # X/y separation, W → kW conversion
│   ├── features/
│   │   └── build_features.py          # physics-informed feature engineering
│   ├── validation/
│   │   └── validate_features.py       # feature schema validation
│   ├── training/
│   │   └── train_model.py             # multi-model training, gate check,
│   │                                  # MLflow registration
│   └── serving/
│       ├── app.py                     # FastAPI — Async telemetry logging,
│       │                              # BackgroundTasks for JSONL creation
│       ├── config.py                  # pydantic-settings, .env support
│       ├── predictor.py               # feature engineering for inference
│       └── streamlit_app.py           # interactive dashboard
│
├── monitoring/                        # Observability Microservice
│   ├── monitor.py                     # Evidently AI — K-S Drift Detection
│   └── reference_data.csv             # Baseline distribution for drift analysis
│
├── logs/                              # Persistent Telemetry
│   └── predictions.jsonl              # Async production inference logs
│
├── reports/                           # Observability Dashboard
│   └── drift_report.html              # Generated Evidently AI dashboard
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
├── airflow/
│   ├── docker-compose.yml             # Airflow stack (local orchestration)
│   ├── Dockerfile.airflow             # Airflow container
│   └── dags/                          # pipeline DAG definitions
│
├── nginx/
│   └── nginx.conf                     # reverse proxy — Routes /report to HTML
│
├── .github/workflows/
│   ├── serve-ci.yml                   # CI — runs 38 tests on every push
│   └── deploy.yml                     # CD — manual deploy to EC2
│
├── dvc.yaml                           # 9-stage DVC pipeline definition
├── dvc.lock                           # pipeline state lock
├── params.yaml                        # training hyperparameters
├── pytest.ini                         # test discovery config
├── requirements.txt                   # pinned production dependencies
├── requirements-dev.txt               # pinned dev/test dependencies
├── Dockerfile.api                     # FastAPI container
├── Dockerfile.monitor                 # 🚀 NEW: Monitoring container
├── Dockerfile.streamlit               # Streamlit container
├── docker-compose.serve.yml           # serving stack — Shared Log Volume
├── docker-compose.serve.prod.yml      # production overrides
├── .env.example                       # config template
├── .gitignore                         # 🚀 UPDATED: Shields secrets/heavy data
└── README.md
```


---
## Quick Start — Local Development

### Prerequisites
- Python 3.10+
- Git
- Docker Desktop

### 1. Clone and install

```bash
git clone https://github.com/beneyalraj/wind_power_ml_project.git
cd wind_power_ml_project

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Pre-configured for local mode (USE_MLFLOW_REGISTRY=false)
# No credentials needed to run locally
```

### 3. Run with Docker (recommended)

```bash
docker compose -f docker-compose.serve.yml up --build
# Dashboard:  http://localhost
# API docs:   http://localhost/api/docs
```

### 4. Run without Docker

```bash
# Terminal 1
uvicorn src.serving.app:app --reload

# Terminal 2
streamlit run src/serving/streamlit_app.py
```

### 5. Run tests

```bash
pytest tests/ -v
# Expected: 38 passed
```

---

## Production Mode — Load Model from Registry

Update `.env` to load directly from DagHub MLflow registry:

```bash
USE_MLFLOW_REGISTRY=true
MLFLOW_TRACKING_URI=https://dagshub.com/beneyalraj/wind_power_system.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

Restart the API — it downloads and loads `WindPowerPredictor@champion` at startup. No code changes required between local and production modes.

---

## API Reference

<here goes your swagger UI image>

| Endpoint | Method | Description |
|---|---|---|
| `GET  /api/health` | GET | Liveness check — model loaded status and version |
| `POST /api/predict` | POST | Single record inference — returns kW with trace ID |
| `POST /api/predict_batch` | POST | Batch inference — up to 500 records, vectorised |
| `GET  /api/docs` | GET | Interactive Swagger UI |

### Example Request

```bash
curl -X POST http://13.219.46.116/api/predict \
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
  "request_id": "b07e1e14-e80f-4114-8341-026329c6d3e2"
}
```

Every response includes a `request_id` (UUID4) attached via middleware and echoed in the `X-Request-ID` response header for end-to-end tracing.

---

## CI/CD Pipeline

### Continuous Integration
Runs automatically on every push to `main` when serving code or tests change.

```
push to main
    ↓
pytest tests/ -v (38 tests)
    ↓
✅ All pass → merge allowed
❌ Any fail → merge blocked
```

### Continuous Deployment
Manual trigger — deploy only when you decide it's ready.

```
GitHub Actions → Deploy to EC2 → Run workflow
    ↓
SSH into EC2
    ↓
git reset --hard origin/main
    ↓
docker-compose down
docker-compose up -d --build
    ↓
docker image prune -af
```

---

## DVC Pipeline

```
data_ingestion → validate_raw → extract_scenario_dataset
    → validate_processed → dataset_split → feature_extraction
    → feature_validation → build_dataset → model_training
```

```bash
dvc repro                    # run full pipeline
dvc repro model_training     # run from specific stage
dvc push                     # push artifacts to DagHub
```

---

## Promoting a Model to Production

After training registers a new model version, promote it to `@champion`:

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
---
## Model Monitoring & Observability
A model is only as good as its last prediction. This project implements a full-cycle monitoring "sidecar" to detect Data Drift and Schema Violation.

Asynchronous Telemetry: Inference inputs and predictions are logged to JSONL files using FastAPI BackgroundTasks. This ensures disk I/O for logging never blocks the high-speed inference response.

Automated Drift Analysis: A scheduled Linux cron job triggers an isolated Evidently AI Docker container.

Statistical Validation: The system performs Kolmogorov-Smirnov (K-S) tests to compare live production data distributions against the training baseline.

Direct Reporting: Nginx is configured to serve the resulting interactive HTML drift reports at the /report endpoint for stakeholder review.

---

## Production Hardening & Reliability

Nginx Reverse Proxy: Acts as a gateway, handling path-based routing for the API (/api), the Dashboard (/), and Monitoring (/report) on a single port (80).

Decoupled Architecture: The monitoring stack is isolated from the serving stack. A failure in the drift analysis container cannot crash the live prediction API.

Shared Volume Strategy: Utilizes Docker Volumes to bridge data between the FastAPI "Producer" and the Monitoring "Consumer" without exposing the internal filesystem.

Enterprise Model Gating: The CI/CD pipeline includes an automated "Performance Gate" ($R^2 \geq 0.85$); models failing this threshold are blocked from the @champion alias in the registry.

---
## Quality Assurance & Testing

100% Endpoint Coverage: 38+ integration tests using HTTPX to validate asynchronous API behavior and response schemas.

Physics-Informed Validation: Unit tests ensure that engineered features (like $v^3$ and cyclical encoding) correctly represent the underlying aerodynamic laws before reaching the model.

---

## Links

- **GitHub:** https://github.com/beneyalraj/wind_power_ml_project
- **DagHub / MLflow:** https://dagshub.com/beneyalraj/wind_power_system
- **Live Dashboard:** http://13.219.46.116
- **Live API Docs:** http://13.219.46.116/api/docs
- **Dataset:** https://registry.opendata.aws/nrel-pds-windai
