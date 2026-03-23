PYTHON := python3.10
VENV := venv
BIN := $(VENV)/bin

.PHONY: venv install init_mlops clean help repro full dev inspect train train-dvc experiment mlflow-ui serve-local ui-local run-local test up-dev up-prod down logs-api logs-ui monitor

# =============================================================================
# MLOps & Data Pipeline Commands
# =============================================================================

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Venv created. Use 'source $(VENV)/bin/activate' to enter."

install: venv
	@echo "Installing dependencies from requirements files..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt

init_mlops:
	@echo "Initializing MLops environment..."
	@if [ ! -d ".git" ]; then git init; fi
	@if [ ! -d ".dvc" ]; then \
		$(BIN)/dvc init; \
		git add .dvc .dvcignore; \
		git commit -m "Initialize Git and DVC"; \
	else \
		echo "DVC is already initialized. Skipping."; \
	fi

repro:
	@echo "Executing DVC pipeline (repro)..."
	$(BIN)/dvc repro

full:
	@echo "Switching to full production mode (501 layouts)..."
	@$(BIN)/python -c "import re; \
		f=open('params.yaml','r+'); \
		text=f.read(); \
		text=re.sub(r'development_mode:\s*true', 'development_mode: false', text); \
		text=re.sub(r'max_layouts:\s*\d+', 'max_layouts: 501', text); \
		f.seek(0); f.write(text); f.truncate(); f.close()"
	@echo "Params updated. Running full pipeline..."
	$(BIN)/dvc repro

dev:
	@echo "Switching back to development mode (10 layouts)..."
	@$(BIN)/python -c "import re; \
		f=open('params.yaml','r+'); \
		text=f.read(); \
		text=re.sub(r'development_mode:\s*false', 'development_mode: true', text); \
		text=re.sub(r'max_layouts:\s*\d+', 'max_layouts: 10', text); \
		f.seek(0); f.write(text); f.truncate(); f.close()"
	@echo "Switched to dev mode. Run 'make repro' when ready."

inspect:
	@echo "Inspecting raw HDF5 data..."
	$(BIN)/python src/ingestion/inspect_h5.py

clean:
	@echo "Cleaning up virtual environment, cache, and compiled files..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# =============================================================================
# Training & Experimentation
# =============================================================================

train:
	@echo "Running model training script natively..."
	$(BIN)/python src/training/train_model.py

train-dvc:
	@echo "Running training via DVC pipeline..."
	$(BIN)/dvc repro model_training

experiment:
	@echo "Running experiment pipeline..."
	$(BIN)/python src/training/train_model.py

mlflow-ui:
	@echo "Starting MLflow UI on http://localhost:5000..."
	$(BIN)/mlflow ui --host 0.0.0.0 --port 5000

# =============================================================================
# Local Development (Non-Dockerized)
# =============================================================================

serve-local:
	@echo "Starting FastAPI server locally on port 8000..."
	$(BIN)/uvicorn src.serving.app:app --reload

ui-local:
	@echo "Starting Streamlit UI locally on port 8501..."
	$(BIN)/streamlit run src/serving/streamlit_app.py

run-local:
	@echo "Starting both FastAPI and Streamlit locally..."
	$(MAKE) serve-local & $(MAKE) ui-local

# =============================================================================
# Serving & Docker Infrastructure Commands
# =============================================================================

test:
	@echo "Running API integration tests..."
	export PYTHONPATH=$$PYTHONPATH:$(PWD) && $(BIN)/pytest tests/ -v --tb=short

up-dev:
	@echo "Starting local development stack (FastAPI + Streamlit + Nginx)..."
	docker compose -f docker-compose.serve.yml up --build

up-prod:
	@echo "Starting production stack (Nginx, log rotation, restart: always)..."
	docker-compose -f docker-compose.serve.yml \
	               -f docker-compose.serve.prod.yml \
	               up -d --build

down:
	@echo "Tearing down all containers and networks..."
	docker-compose -f docker-compose.serve.yml \
	               -f docker-compose.serve.prod.yml \
	               down

logs-api:
	@echo "Tailing FastAPI container logs..."
	docker logs -f wind-api

logs-ui:
	@echo "Tailing Streamlit container logs..."
	docker logs -f wind-streamlit

monitor:
	@echo "Running Evidently drift detection..."
	@echo "Report will be available at http://13.219.46.116/report"
	docker-compose -f docker-compose.serve.yml \
	               --profile manual run --rm monitor

# =============================================================================
# Help Menu
# =============================================================================

help:
	@echo "====================================================================="
	@echo "Wind Power System — Available Commands"
	@echo "====================================================================="
	@echo ""
	@echo "Data & ML Pipeline:"
	@echo "  make venv        : Create virtual environment"
	@echo "  make install     : Install all dependencies"
	@echo "  make init_mlops  : Initialize Git and DVC (idempotent)"
	@echo "  make repro       : Run full DVC pipeline"
	@echo "  make full        : Switch to production mode (501 layouts) + repro"
	@echo "  make dev         : Switch back to development mode (10 layouts)"
	@echo "  make inspect     : Inspect raw HDF5 data structure"
	@echo "  make clean       : Remove venv, cache, compiled files"
	@echo ""
	@echo "Training & Tracking:"
	@echo "  make train       : Run training script natively"
	@echo "  make train-dvc   : Run training stage via DVC"
	@echo "  make mlflow-ui   : Start MLflow UI on http://localhost:5000"
	@echo ""
	@echo "Local Dev (Python — no Docker):"
	@echo "  make serve-local : Run FastAPI with uvicorn (hot-reload)"
	@echo "  make ui-local    : Run Streamlit dashboard"
	@echo "  make run-local   : Run both API and UI locally"
	@echo ""
	@echo "Serving & Ops (Docker):"
	@echo "  make test        : Run full test suite (38 tests)"
	@echo "  make up-dev      : Start local dev stack (with hot-reload)"
	@echo "  make up-prod     : Start production stack (detached, Nginx)"
	@echo "  make down        : Stop and remove all containers"
	@echo "  make logs-api    : Tail FastAPI backend logs"
	@echo "  make logs-ui     : Tail Streamlit frontend logs"
	@echo "  make monitor     : Run Evidently drift detection manually"
	@echo ""
	@echo "====================================================================="