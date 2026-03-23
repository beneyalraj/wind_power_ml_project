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

## Table of Contents

- [Live Demo](#-live-demo)
- [Business Context](#business-context)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Orchestration — Apache Airflow](#orchestration--apache-airflow)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Model Monitoring & Observability](#model-monitoring--observability)
- [Engineered Features](#engineered-features)
- [Project Structure](#project-structure)
- [Quick Start — Local Development](#quick-start--local-development)
- [Production Mode](#production-mode--load-model-from-registry)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)
- [DVC Pipeline](#dvc-pipeline)
- [Promoting a Model to Production](#promoting-a-model-to-production)
- [Quality Assurance & Testing](#quality-assurance--testing)
- [Links](#links)

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| Streamlit Dashboard | http://13.219.46.116 |
| FastAPI Swagger UI | http://13.219.46.116/api/docs |
| Health Check | http://13.219.46.116/api/health |
| Drift Report | http://13.219.46.116/report |

*Streamlit dashboard showing live prediction with power curve analysis:*

<img width="1433" height="811" alt="Streamlit dashboard with prediction result and power curve" src="https://github.com/user-attachments/assets/6ad4dafa-8ee2-45b2-bcbd-23bc0cafb06c" />

*System information expander showing ML stack, pipeline steps, and model performance:*

<img width="1433" height="695" alt="Streamlit system information panel" src="https://github.com/user-attachments/assets/bed3077f-2550-4767-b2b7-994d2345e9f0" />

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
| **ML & Data** | scikit-learn, pandas, numpy, scipy, Pandera, joblib |
| **MLOps** | DVC, MLflow, DagHub, Evidently AI |
| **Serving** | FastAPI, Pydantic v2, Uvicorn, Streamlit |
| **Infrastructure** | Docker, Nginx, AWS EC2, GitHub Actions |
| **Orchestration** | Apache Airflow, Linux cron |
| **Storage** | AWS S3, DagHub remote |
| **Language** | Python 3.10 |
| **Testing** | pytest, httpx |

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
│   Docker Compose (AWS EC2 t3.micro)                             │
│   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│   │    Nginx     │  │    FastAPI      │  │   Streamlit      │  │
│   │   :80        │  │   :8000         │  │   :8501          │  │
│   │              │  │                 │  │                  │  │
│   │ /api/* ────► │  │ POST /predict   │◄─│ Input sliders    │  │
│   │ /* ───────►  │  │ POST /predict   │  │ Power curve      │  │
│   │ /report ──┐  │  │   _batch        │  │ System info      │  │
│   │ (HTML)    │  │  │ GET  /health    │  │                  │  │
│   │           │  │  │ BackgroundTasks │  │ API_URL=         │  │
│   │ Reverse   │  │  │ (async logging) │  │ http://fastapi   │  │
│   │ Proxy     │  │  └────────┬────────┘  │ :8000            │  │
│   └─────▲─────┘             │           └──────────────────┘  │
│         │                   │ (JSONL logs)                     │
│         │                   ▼                                  │
│         │     ┌─────────────────────────────┐                  │
│         │     │  Shared Docker Volume       │                  │
│         │     │  logs/predictions.jsonl     │                  │
│         │     └──────────────┬──────────────┘                  │
│         │                    │                                  │
│         │                    ▼                                  │
│         │     ┌─────────────────────────────┐                  │
│         └─────┤  Evidently AI Container     │                  │
│               │  profiles: manual           │                  │
│               │  cron: weekly @ 2am Mon     │                  │
│               └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Orchestration — Apache Airflow

The training pipeline is orchestrated locally using Apache Airflow. Airflow manages task dependencies, retries, and scheduling — triggering the DVC pipeline stages in the correct order.

*Airflow DAG view showing the wind power training pipeline:*

<img width="1427" height="799" alt="Screenshot 2026-03-21 at 9 18 12 PM" src="https://github.com/user-attachments/assets/8b1163a4-257e-4643-b38f-db66bfa16ed6" />


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

*MLflow experiment tracking showing multiple model runs and metrics comparison:*

<img width="1440" height="811" alt="MLflow experiments showing model runs and metrics" src="https://github.com/user-attachments/assets/0c09074a-a301-4fff-b740-89710b311609" />

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

## Model Monitoring & Observability

A model is only as good as its last prediction. This project implements production drift monitoring to detect when the model's input distribution shifts away from its training baseline.

**How it works:**

1. **Async telemetry** — every prediction input and output is logged to `logs/predictions.jsonl` using FastAPI `BackgroundTasks`. The disk write happens after the response is returned — zero latency impact on inference.

2. **Drift detection** — an isolated Evidently AI Docker container reads the prediction log, compares the live distribution against the training baseline using Kolmogorov-Smirnov tests, and generates an interactive HTML report.

3. **Report serving** — Nginx serves the report directly at `/report`. No SSH required — accessible to anyone with the URL.

4. **Scheduled** — a Linux cron job on EC2 triggers the monitor container every Monday at 2am automatically.

*Evidently AI drift report showing feature distribution comparison:*

<img width="1424" height="811" alt="Evidently AI drift report showing feature distributions" src="https://github.com/user-attachments/assets/02f365e9-370f-4fb0-8ebf-e51bba4ab356" />

**Run monitoring manually:**

```bash
# Trigger the monitor container on EC2
docker-compose -f docker-compose.serve.yml \
               --profile manual run --rm monitor

# View the report
open http://13.219.46.116/report
```

**Cron schedule (EC2):**
```bash
# Runs every Monday at 2am
0 2 * * 1 cd ~/wind_power_ml_project && \
  docker-compose -f docker-compose.serve.yml \
  --profile manual run --rm monitor \
  >> logs/monitor_cron.log 2>&1
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
│       ├── app.py                     # FastAPI — endpoints, BackgroundTasks
│       │                              # async logging, dual registry loading
│       ├── config.py                  # pydantic-settings, .env support
│       ├── predictor.py               # feature engineering for inference
│       └── streamlit_app.py           # interactive dashboard
│
├── monitoring/
│   ├── monitor.py                     # Evidently AI — K-S drift detection
│   └── reference_data.csv             # training baseline for drift analysis
│
├── logs/
│   └── predictions.jsonl              # async prediction log (gitignored)
│
├── reports/
│   └── drift_report.html              # generated Evidently report (gitignored)
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
│   └── nginx.conf                     # reverse proxy — /api, /, /report routing
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
├── Dockerfile.streamlit               # Streamlit container
├── Dockerfile.monitor                 # Evidently monitoring container
├── docker-compose.serve.yml           # serving stack definition
├── docker-compose.serve.prod.yml      # production overrides (log rotation,
│                                      # restart: always)
├── .env.example                       # config template
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
# Local development
docker compose -f docker-compose.serve.yml up --build
# Dashboard:  http://localhost
# API docs:   http://localhost/api/docs

# Production mode (EC2)
docker-compose -f docker-compose.serve.yml \
               -f docker-compose.serve.prod.yml \
               up -d --build
```

### 4. Run without Docker

```bash
# Terminal 1 — API
uvicorn src.serving.app:app --reload

# Terminal 2 — Dashboard
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

*FastAPI Swagger UI showing interactive API documentation:*

<img width="1426" height="811" alt="FastAPI Swagger UI with interactive endpoints" src="https://github.com/user-attachments/assets/fc0397f5-dc36-48e9-b95f-873715e5641c" />

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

*GitHub Actions CI showing all tests passing:*

<img width="1440" height="812" alt="GitHub Actions CI pipeline passing" src="https://github.com/user-attachments/assets/b4f6d3d5-193e-4bb5-8530-03696f3b8a41" />

<img width="1425" height="813" alt="GitHub Actions CI test results" src="https://github.com/user-attachments/assets/48e0285c-1eae-46b4-a159-d402fccd56f1" />

```
push to main
    ↓
pytest tests/ -v (38 tests)
    ↓
✅ All pass → merge allowed
❌ Any fail → merge blocked
```

### Continuous Deployment

Manual trigger only — deploy when you decide it's ready.

*GitHub Actions CD manual deployment to EC2:*

<img width="1440" height="732" alt="GitHub Actions CD deployment to AWS EC2" src="https://github.com/user-attachments/assets/fd42f1dd-3214-4b6e-9c44-0efcc31a1f9f" />

```
GitHub Actions → Deploy to EC2 → Run workflow
    ↓
SSH into EC2
    ↓
git reset --hard origin/main
    ↓
docker-compose down
docker-compose up -d --build --remove-orphans
    ↓
docker image prune -af
```

---

## DVC Pipeline

The 9-stage training pipeline is defined as code in `dvc.yaml`. DVC tracks data versions, caches intermediate outputs, and only reruns stages whose dependencies have changed.

*DVC pipeline DAG showing stage dependencies:*

<img width="1323" height="773" alt="Screenshot 2026-03-21 at 9 07 09 PM" src="https://github.com/user-attachments/assets/f3c96645-2d0c-4dfb-b4cd-8f7b9439d016" />


```
data_ingestion → validate_raw → extract_scenario_dataset
    → validate_processed → dataset_split → feature_extraction
    → feature_validation → build_dataset → model_training
```

```bash
dvc repro                    # run full pipeline
dvc repro model_training     # run from specific stage only
dvc push                     # push artifacts to DagHub remote
```

---

## Promoting a Model to Production

After training registers a new model version, promote it to the `@champion` alias:

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

## Quality Assurance & Testing

The test suite covers three critical areas:

**API integration tests (27 tests — `tests/test_api.py`)**
- Happy path: valid request returns 200, correct response schema, positive float prediction
- Request tracing: `X-Request-ID` header present on every response, body ID matches header
- Domain constraints: negative model output is clipped to exactly 0.0 (regression guard)
- Pydantic validation: all five field constraints enforced (negative wind speed, direction > 360, etc.)
- Batch endpoint: count parity, non-negative outputs, empty/oversized batch guards

**Predictor unit tests (11 tests — `tests/test_predictor.py`)**
- Mathematical correctness of each engineered feature (sin/cos values, v², v³, wake formula)
- Feature ordering matches training order — a silent reordering produces wrong predictions
- Edge cases: zero wind speed, north wind direction, missing feature names

**Test isolation**
- All tests run without real model files — `conftest.py` mocks the model, filesystem, and MLflow registry
- Two separate fixtures: `client` (happy path, 5000 kW output) and `negative_output_client` (−999 kW, tests domain clipping)
- `scope="module"` means the app boots once per test module — mirrors production startup cost

```bash
pytest tests/ -v
# 38 passed
```

---

## Links

- **GitHub:** https://github.com/beneyalraj/wind_power_ml_project
- **DagHub / MLflow:** https://dagshub.com/beneyalraj/wind_power_system
- **Live Dashboard:** http://13.219.46.116
- **Live API Docs:** http://13.219.46.116/api/docs
- **Drift Report:** http://13.219.46.116/report
- **Dataset:** https://registry.opendata.aws/nrel-pds-windai
