PYTHON := python3.10
VENV := venv
BIN := $(VENV)/bin

.PHONY: venv install init_mlops clean help repro full

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Venv created. Use 'source $(VENV)/bin/activate' to enter."

install: venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

init_mlops:
	git init
	# Ensure DVC is initialized only if it hasn't been yet
	$(BIN)/dvc init
	git add .dvc .dvcignore
	git commit -m "Initialize Git and DVC"

repro:
	$(BIN)/dvc repro

full:
	@echo "Switching to full production mode..."
	@sed -i '' 's/development_mode: true/development_mode: false/' params.yaml
	$(BIN)/dvc repro


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
	@echo "  make repro       : Run full DVC pipeline"
	@echo "  make full        : Run full production extraction"
	@echo "  make inspect     : Inspect raw HDF5"
	@echo "  make clean       : Clean environment"