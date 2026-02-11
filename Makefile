PYTHON := python3.10
VENV := venv
BIN := $(VENV)/bin

.PHONY: venv install init_mlops clean help

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Venv created. Use 'source $(VENV)/bin/activate' to enter."

install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

init_mlops:
	git init
	# Ensure DVC is initialized only if it hasn't been yet
	$(BIN)/dvc init --no-scm
	git add .dvc .dvcignore
	git commit -m "Initialize Git and DVC"

download:
	$(BIN)/python src/data/load_data.py

validate-raw:
	$(BIN)/python src/data/validate_raw.py

inspect:
	$(BIN)/python src/data/inspect_h5.py


clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make venv        : Create virtual environment"
	@echo "  make install     : Install requirements"
	@echo "  make init_mlops  : Initialize Git and DVC"
	@echo "  make download     : Download data from S3"
	@echo "  make validate-raw : Validate raw HDF5 data"
	@echo "  make inspect      : Inspect HDF5 file structure"
	@echo "  make clean       : Remove venv and temp files"