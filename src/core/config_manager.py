"""
Centralized configuration and schema management.

Responsibilities:
- Load extraction configuration
- Load dataset schema contract
- Validate structure and required fields
- Fail fast on malformed config/schema
"""

from pathlib import Path
from typing import Dict, Any
import yaml


class ConfigManager:
    """
    Handles loading and validating runtime configuration
    and dataset schema contracts.
    """

    REQUIRED_EXTRACTION_KEYS = [
        "input_path",
        "output_path",
        "schema_path",
        "development_mode",
        "layouts_per_iteration",
        "rows_per_batch",
        "partition_by"
    ]

    REQUIRED_SCHEMA_SECTIONS = [
        "identifiers",
        "features",
        "target"
    ]

    def __init__(self,
                 config_path: str = "params.yaml"):

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_yaml(self.config_path)
        self.extraction_config = self._validate_extraction_config()

        self.schema_path = Path(self.extraction_config["schema_path"])
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        self.schema = self._validate_schema()


    # YAML Loading


    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)

    # Extraction Config Validation
    

    def _validate_extraction_config(self) -> Dict[str, Any]:

        if "extraction" not in self.config:
            raise KeyError("'extraction' section missing in config")

        extraction = self.config["extraction"]

        for key in self.REQUIRED_EXTRACTION_KEYS:
            if key not in extraction:
                raise KeyError(f"Missing required extraction key: {key}")

        # Validate numeric parameters
        if extraction["rows_per_batch"] <= 0:
            raise ValueError("rows_per_batch must be > 0")

        if extraction["layouts_per_iteration"] <= 0:
            raise ValueError("layouts_per_iteration must be > 0")

        if not isinstance(extraction["partition_by"], list):
            raise ValueError("partition_by must be a list")

        # Validate input file exists
        input_path = Path(extraction["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        return extraction

    
    # Schema Validation
    

    def _validate_schema(self) -> Dict[str, Any]:

        schema = self._load_yaml(self.schema_path)

        for section in self.REQUIRED_SCHEMA_SECTIONS:
            if section not in schema:
                raise KeyError(f"Schema missing required section: {section}")

        # Validate identifiers
        if not isinstance(schema["identifiers"], dict):
            raise ValueError("identifiers must be a dictionary")

        # Validate features
        if not isinstance(schema["features"], dict):
            raise ValueError("features must be a dictionary")

        # Validate target
        if not isinstance(schema["target"], dict):
            raise ValueError("target must be a dictionary")

        # Ensure raw_source and dtype exist for features
        for feature_name, meta in schema["features"].items():
            if "raw_source" not in meta:
                raise KeyError(f"{feature_name} missing raw_source")
            if "dtype" not in meta:
                raise KeyError(f"{feature_name} missing dtype")

        # Validate target metadata
        target_meta = next(iter(schema["target"].values()))
        if "raw_source" not in target_meta:
            raise KeyError("Target missing raw_source")
        if "dtype" not in target_meta:
            raise KeyError("Target missing dtype")

        return schema


    def get_extraction_config(self) -> Dict[str, Any]:
        return self.extraction_config

    def get_schema(self) -> Dict[str, Any]:
        return self.schema
