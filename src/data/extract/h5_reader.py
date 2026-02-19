"""
Scenario-level HDF5 streaming reader.

Grain:
One row per (layout_id, scenario_id)

Responsibilities:
- Deterministic iteration over layouts and scenarios
- Schema-driven raw field mapping
- Schema-driven aggregation
- Constraint enforcement
- Memory-efficient streaming
"""

import h5py
from typing import Iterator, Dict, Any
import logging

from src.core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class H5StreamingReader:

    def __init__(self, config_path: str = "params.yaml"):

        self.cm = ConfigManager(config_path)
        self.config = self.cm.get_extraction_config()
        self.schema = self.cm.get_schema()

        self.h5_path = self.config["input_path"]
        self.layouts_per_iteration = self.config["layouts_per_iteration"]
        self.dev_mode = self.config["development_mode"]
        self.max_layouts = self.config.get("max_layouts")

    # =====================================================
    # Public Iterator
    # =====================================================

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """
        Yields one record per scenario (scenario-level grain).
        """

        with h5py.File(self.h5_path, "r") as f:

            layouts = sorted(f.keys())

            if self.dev_mode and self.max_layouts:
                layouts = layouts[:self.max_layouts]

            for i in range(0, len(layouts), self.layouts_per_iteration):

                layout_batch = layouts[i:i + self.layouts_per_iteration]

                for layout_name in layout_batch:
                    yield from self._process_layout(f, layout_name)

    # =====================================================
    # Layout Processing
    # =====================================================

    def _process_layout(self, f, layout_name: str):

        layout = f[layout_name]

        if "Scenarios" not in layout:
            logger.warning(f"{layout_name} missing Scenarios group")
            return

        scenarios = sorted(layout["Scenarios"].keys())

        for scenario_name in scenarios:
            scenario = layout["Scenarios"][scenario_name]

            yield self._process_scenario(
                scenario,
                layout_name,
                scenario_name
            )

    # =====================================================
    # Scenario Processing (Scenario-Level)
    # =====================================================

    def _process_scenario(
        self,
        scenario,
        layout_name: str,
        scenario_name: str
    ) -> Dict[str, Any]:

        identifiers_schema = self.schema["identifiers"]
        features_schema = self.schema["features"]
        target_schema = self.schema["target"]

        record = {}

        # -------------------------------------------------
        # Identifiers (Derived from structure)
        # -------------------------------------------------

        for id_name, meta in identifiers_schema.items():

            raw_source = meta["raw_source"]

            if raw_source == "derived:layout_name":
                value = layout_name

            elif raw_source == "derived:scenario_name":
                value = scenario_name

            else:
                raise ValueError(
                    f"Unsupported identifier raw_source: {raw_source}"
                )

            record[id_name] = self._enforce_constraints(
                value,
                meta,
                id_name
            )

        # -------------------------------------------------
        # Features
        # -------------------------------------------------

        for feature_name, meta in features_schema.items():

            raw_source = meta["raw_source"]

            if raw_source.startswith("derived:"):
                value = self._compute_derived_feature(
                    raw_source,
                    scenario,
                    target_schema
                )

            else:
                if raw_source not in scenario:
                    raise KeyError(f"Missing raw field: {raw_source}")

                value = scenario[raw_source][()]

            record[feature_name] = self._enforce_constraints(
                value,
                meta,
                feature_name
            )

        # -------------------------------------------------
        # Target (Schema-Driven Aggregation)
        # -------------------------------------------------

        target_name, target_meta = next(iter(target_schema.items()))

        raw_target_field = target_meta["raw_source"]
        aggregation = target_meta.get("aggregation")

        if raw_target_field not in scenario:
            raise KeyError(f"Missing target field: {raw_target_field}")

        raw_values = scenario[raw_target_field][:]

        target_value = self._apply_aggregation(
            raw_values,
            aggregation
        )

        record[target_name] = self._enforce_constraints(
            target_value,
            target_meta,
            target_name
        )

        return record

    # =====================================================
    # Derived Feature Logic
    # =====================================================

    def _compute_derived_feature(
        self,
        raw_source: str,
        scenario,
        target_schema
    ):

        if raw_source == "derived:num_turbines":
            target_field = next(iter(target_schema.values()))["raw_source"]
            return len(scenario[target_field][:])

        raise ValueError(f"Unsupported derived feature: {raw_source}")

    # =====================================================
    # Aggregation Logic
    # =====================================================

    def _apply_aggregation(self, values, aggregation: str):

        if aggregation == "sum":
            return float(values.sum())

        elif aggregation == "mean":
            return float(values.mean())

        elif aggregation == "max":
            return float(values.max())

        elif aggregation == "min":
            return float(values.min())

        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation}")

    # =====================================================
    # Constraint Enforcement
    # =====================================================

    def _enforce_constraints(
        self,
        value: Any,
        meta: Dict[str, Any],
        field_name: str
    ):

        dtype = meta.get("dtype")

        if dtype == "float32":
            value = float(value)

        elif dtype == "int32":
            value = int(value)

        elif dtype == "string":
            value = str(value)

        # Nullability
        if not meta.get("nullable", False) and value is None:
            raise ValueError(f"{field_name} cannot be null")

        # Min constraint
        if "min" in meta and value < meta["min"]:
            raise ValueError(f"{field_name} below min constraint")

        # Max constraint
        if "max" in meta and value > meta["max"]:
            raise ValueError(f"{field_name} above max constraint")

        return value
