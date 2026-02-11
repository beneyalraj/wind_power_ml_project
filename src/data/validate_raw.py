
import os
import re
import sys
import yaml
import h5py
import logging
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Setup logging
os.makedirs("logs", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/raw_validation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ValidationReport:
    """Collects validation results and generates reports"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}
        self.validation_time = None
        self.start_time = datetime.now()
    
    def add_error(self, error: str, context: str = ""):
        """Add validation error"""
        self.errors.append({
            "error": error,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"[{context}] {error}")
    
    def add_warning(self, warning: str, context: str = ""):
        """Add validation warning"""
        self.warnings.append({
            "warning": warning,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        logger.warning(f"[{context}] {warning}")
    
    def add_stat(self, key: str, value: Any):
        """Add statistic"""
        self.stats[key] = value
        logger.info(f"Stat: {key} = {value}")
    
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0
    
    def finalize(self):
        """Calculate final metrics"""
        self.validation_time = (datetime.now() - self.start_time).total_seconds()
    
    def save(self, path: str = "metrics/raw_validation.json"):
        """Save validation report as JSON"""
        self.finalize()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "PASS" if self.is_valid() else "FAIL",
            "validation_time_seconds": round(self.validation_time, 2),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.stats
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Validation report saved to {path}")
        return report


class RawDataValidator:
    """Production-grade HDF5 validator with memory-efficient processing"""
    
    def __init__(self, config_path: str = "params.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.contract = self.config.get("raw_validation", {})
        self.file_path = self.config['data_ingestion']['local_path']
        self.report = ValidationReport()
        
        # Validation settings
        self.sample_size = self.contract.get("sample_size", 10)
        self.chunk_size = self.contract.get("chunk_size", 10000)
        
        logger.info(f"Validator initialized: sample_size={self.sample_size}, chunk_size={self.chunk_size}")
    
    def validate_all(self) -> ValidationReport:
        """Run complete validation pipeline"""
        logger.info("="*70)
        logger.info("STARTING RAW DATA VALIDATION")
        logger.info("="*70)
        logger.info(f"File: {self.file_path}")
        logger.info("")
        
        try:
            # Phase 1: File-level validation
            self._validate_file()
            
            # Phase 2: Structure and data validation
            if self.report.is_valid():  # Only continue if file validation passed
                with h5py.File(self.file_path, "r") as f:
                    # Discover layouts
                    layouts = self._discover_layouts(f)
                    
                    if layouts:
                        # Sample for validation (don't validate all!)
                        sample_layouts = layouts[:self.sample_size]
                        logger.info(f"📊 Validating {len(sample_layouts)} sample layouts (out of {len(layouts)} total)")
                        
                        # Validate each sampled layout
                        for i, layout_name in enumerate(sample_layouts, 1):
                            logger.info(f"\n--- Layout {i}/{len(sample_layouts)}: {layout_name} ---")
                            self._validate_layout(f, layout_name)
                        
                        # Phase 3: Data profiling
                        self._profile_data(f, sample_layouts)
        
        except FileNotFoundError as e:
            self.report.add_error(f"File not found: {str(e)}", "CRITICAL")
        except Exception as e:
            self.report.add_error(f"Unexpected error: {str(e)}", "CRITICAL")
            logger.exception("Validation failed with exception:")
        
        # Finalize report (calculate validation time)
        self.report.finalize()
        
        # Generate summary
        self._log_summary()
        
        # Save report
        self.report.save()
        
        return self.report
    
    def _validate_file(self):
        """Validate file-level requirements"""
        logger.info("Phase 1: File-level validation")
        logger.info("-" * 50)
        
        contract = self.contract.get("file", {})
        
        # Check existence
        if contract.get("must_exist", True):
            if not os.path.exists(self.file_path):
                self.report.add_error(f"File does not exist: {self.file_path}", "FILE")
                return
        
        logger.info(f"✅ File exists: {self.file_path}")
        
        # Check size
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        self.report.add_stat("file_size_mb", round(file_size_mb, 2))
        
        min_size = contract.get("min_size_mb", 0)
        if file_size_mb < min_size:
            self.report.add_error(
                f"File size {file_size_mb:.2f}MB is below minimum {min_size}MB",
                "FILE"
            )
        else:
            logger.info(f"✅ File size: {file_size_mb:.2f} MB (minimum: {min_size} MB)")
        
        # Check if file is valid HDF5
        try:
            with h5py.File(self.file_path, 'r') as f:
                logger.info(f"✅ File is valid HDF5 format")
        except Exception as e:
            self.report.add_error(f"File is not valid HDF5: {str(e)}", "FILE")
    
    def _discover_layouts(self, h5_file) -> List[str]:
        """Discover and validate layouts"""
        logger.info("\nPhase 2: Layout discovery")
        logger.info("-" * 50)
        
        contract = self.contract.get("layout", {})
        pattern = re.compile(contract.get("name_pattern", "^Layout"))
        
        # Find matching layouts
        all_keys = list(h5_file.keys())
        layouts = [key for key in all_keys if pattern.match(key)]
        
        self.report.add_stat("total_keys", len(all_keys))
        self.report.add_stat("total_layouts", len(layouts))
        
        logger.info(f"Total keys in file: {len(all_keys)}")
        logger.info(f"Layouts matching pattern '{pattern.pattern}': {len(layouts)}")
        
        # Check minimum layouts
        min_layouts = contract.get("min_layouts", 1)
        if len(layouts) < min_layouts:
            self.report.add_error(
                f"Only {len(layouts)} layouts found, minimum required: {min_layouts}",
                "STRUCTURE"
            )
        else:
            logger.info(f"✅ Found {len(layouts)} layouts (minimum: {min_layouts})")
        
        return layouts
    
    def _validate_layout(self, h5_file, layout_name: str):
        """Validate a single layout"""
        layout_group = h5_file[layout_name]
        
        # Check if layout has Scenarios group
        if 'Scenarios' not in layout_group:
            self.report.add_error(
                f"Layout missing 'Scenarios' group",
                layout_name
            )
            return
        
        scenarios_group = layout_group['Scenarios']
        scenarios = list(scenarios_group.keys())
        
        if not scenarios:
            self.report.add_error("No scenarios found in layout", layout_name)
            return
        
        logger.info(f"  Found {len(scenarios)} scenarios")
        
        # Validate first scenario (representative sample)
        first_scenario = scenarios_group[scenarios[0]]
        logger.info(f"  Validating scenario: {scenarios[0]}")
        
        # Structure validation
        self._validate_structure(first_scenario, f"{layout_name}/{scenarios[0]}")
        
        # Data type validation
        self._validate_dtypes(first_scenario, f"{layout_name}/{scenarios[0]}")
        
        # Dimension validation
        self._validate_dimensions(first_scenario, f"{layout_name}/{scenarios[0]}")
        
        # Constraint validation (chunked)
        self._validate_constraints(first_scenario, f"{layout_name}/{scenarios[0]}")
    
    def _validate_structure(self, scenario_group, context: str):
        """Validate required datasets exist"""
        required = self.contract.get("required_datasets", [])
        
        missing = []
        for ds in required:
            if ds not in scenario_group:
                missing.append(ds)
        
        if missing:
            self.report.add_error(
                f"Missing required datasets: {', '.join(missing)}",
                context
            )
        else:
            logger.info(f"  ✅ All required datasets present")
    
    def _validate_dtypes(self, scenario_group, context: str):
        """Validate data types"""
        expected_dtypes = self.contract.get("dtypes", {})
        
        dtype_errors = []
        for ds, expected in expected_dtypes.items():
            if ds in scenario_group:
                actual = str(scenario_group[ds].dtype)
                if actual != expected:
                    dtype_errors.append(f"{ds} (expected {expected}, got {actual})")
        
        if dtype_errors:
            self.report.add_error(
                f"Dtype mismatches: {', '.join(dtype_errors)}",
                context
            )
        else:
            logger.info(f"  ✅ Data types validated")
    
    def _validate_dimensions(self, scenario_group, context: str):
        """Validate dataset dimensions"""
        dims_config = self.contract.get("dimensions", {})
        required = self.contract.get("required_datasets", [])
        
        for ds in required:
            if ds not in scenario_group:
                continue
            
            shape = scenario_group[ds].shape
            
            # Check non-empty
            if dims_config.get("require_non_empty", True):
                if np.prod(shape) == 0:
                    self.report.add_error(
                        f"Dataset '{ds}' is empty (shape: {shape})",
                        context
                    )
                    continue
            
            logger.info(f"    {ds}: shape={shape}, dtype={scenario_group[ds].dtype}")
    
    def _validate_constraints(self, scenario_group, context: str):
        """Validate data constraints with chunked processing (memory-efficient)"""
        constraints = self.contract.get("constraints", {})
        
        for ds, rules in constraints.items():
            if ds not in scenario_group:
                self.report.add_warning(f"Dataset '{ds}' not found for constraint validation", context)
                continue
            
            dataset = scenario_group[ds]
            shape = dataset.shape
            
            # Handle scalar vs array datasets
            if shape == ():
                # Scalar value
                value = dataset[()]
                self._validate_scalar_value(ds, value, rules, context)
            else:
                # Array - use chunked processing
                self._validate_array_chunked(ds, dataset, rules, context)
    
    def _validate_scalar_value(self, ds_name: str, value: float, rules: Dict, context: str):
        """Validate a scalar value against constraints"""
        # NaN check
        if not rules.get("allow_nan", True):
            if np.isnan(value):
                self.report.add_error(f"NaN detected in {ds_name}", context)
                return
        
        # Inf check
        if not rules.get("allow_inf", True):
            if np.isinf(value):
                self.report.add_error(f"Infinite value detected in {ds_name}", context)
                return
        
        # Range checks
        if "min" in rules and value < rules["min"]:
            self.report.add_error(
                f"{ds_name} value {value} below minimum {rules['min']}",
                context
            )
        
        if "max" in rules and value > rules["max"]:
            self.report.add_error(
                f"{ds_name} value {value} above maximum {rules['max']}",
                context
            )
    
    def _validate_array_chunked(self, ds_name: str, dataset, rules: Dict, context: str):
        """Validate array dataset using memory-efficient chunking"""
        total_size = dataset.shape[0]
        
        # Process in chunks to avoid memory issues
        start = 0
        nan_found = False
        inf_found = False
        range_violations = 0
        
        while start < total_size:
            end = min(start + self.chunk_size, total_size)
            
            try:
                # Load chunk (only this chunk is in memory)
                chunk = dataset[start:end]
                
                # NaN check
                if not rules.get("allow_nan", True) and not nan_found:
                    if np.isnan(chunk).any():
                        self.report.add_error(f"NaN values detected in {ds_name}", context)
                        nan_found = True
                
                # Inf check
                if not rules.get("allow_inf", True) and not inf_found:
                    if np.isinf(chunk).any():
                        self.report.add_error(f"Infinite values detected in {ds_name}", context)
                        inf_found = True
                
                # Range checks
                if "min" in rules:
                    violations = (chunk < rules["min"]).sum()
                    if violations > 0:
                        range_violations += violations
                
                if "max" in rules:
                    violations = (chunk > rules["max"]).sum()
                    if violations > 0:
                        range_violations += violations
                
                # Chunk is automatically freed when going out of scope
                del chunk
            
            except Exception as e:
                self.report.add_error(
                    f"Error validating {ds_name} chunk [{start}:{end}]: {str(e)}",
                    context
                )
                break
            
            start = end
        
        # Report range violations as warning (might be acceptable)
        if range_violations > 0:
            self.report.add_warning(
                f"{ds_name} has {range_violations} values outside expected range",
                context
            )
    
    def _profile_data(self, h5_file, sample_layouts: List[str]):
        """Generate data statistics and profiling information"""
        logger.info("\nPhase 3: Data profiling")
        logger.info("-" * 50)
        
        if not sample_layouts:
            logger.warning("No layouts to profile")
            return
        
        # Profile first layout
        first_layout = h5_file[sample_layouts[0]]
        
        if 'Scenarios' not in first_layout:
            return
        
        scenarios = list(first_layout['Scenarios'].keys())
        self.report.add_stat("scenarios_per_layout", len(scenarios))
        logger.info(f"Scenarios per layout: {len(scenarios)}")
        
        if not scenarios:
            return
        
        # Profile first scenario
        first_scenario = first_layout['Scenarios'][scenarios[0]]
        
        # Count fields
        fields = list(first_scenario.keys())
        self.report.add_stat("fields_per_scenario", len(fields))
        logger.info(f"Fields per scenario: {len(fields)}")
        
        # Sample scalar values
        scalar_samples = {}
        array_shapes = {}
        
        for key in fields:
            dataset = first_scenario[key]
            shape = dataset.shape
            
            if shape == ():
                # Scalar
                scalar_samples[key] = float(dataset[()])
            else:
                # Array
                array_shapes[key] = shape
                if key == 'Turbine Power':
                    self.report.add_stat("num_turbines", shape[0])
        
        self.report.add_stat("scalar_samples", scalar_samples)
        self.report.add_stat("array_shapes", {k: list(v) for k, v in array_shapes.items()})
        
        logger.info(f"\nScalar values (sample):")
        for key, value in scalar_samples.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\nArray shapes (sample):")
        for key, shape in array_shapes.items():
            logger.info(f"  {key}: {shape}")
    
    def _log_summary(self):
        """Log validation summary"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        
        status = "✅ PASS" if self.report.is_valid() else "❌ FAIL"
        logger.info(f"Status: {status}")
        logger.info(f"Errors: {len(self.report.errors)}")
        logger.info(f"Warnings: {len(self.report.warnings)}")
        logger.info(f"Validation time: {self.report.validation_time:.2f}s")
        
        if self.report.errors:
            logger.error("\n❌ ERRORS:")
            for i, err in enumerate(self.report.errors[:10], 1):
                logger.error(f"  {i}. [{err['context']}] {err['error']}")
            if len(self.report.errors) > 10:
                logger.error(f"  ... and {len(self.report.errors) - 10} more errors")
        
        if self.report.warnings:
            logger.warning("\n⚠️  WARNINGS:")
            for i, warn in enumerate(self.report.warnings[:10], 1):
                logger.warning(f"  {i}. [{warn['context']}] {warn['warning']}")
            if len(self.report.warnings) > 10:
                logger.warning(f"  ... and {len(self.report.warnings) - 10} more warnings")
        
        if self.report.stats:
            logger.info("\n📊 KEY STATISTICS:")
            key_stats = ['file_size_mb', 'total_layouts', 'scenarios_per_layout', 'num_turbines']
            for stat in key_stats:
                if stat in self.report.stats:
                    logger.info(f"  {stat}: {self.report.stats[stat]}")


def main():
    """Entry point for DVC pipeline"""
    try:
        validator = RawDataValidator()
        report = validator.validate_all()
        
        if report.is_valid():
            logger.info("\n🎉 Validation completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n💥 Validation failed - see errors above")
            sys.exit(1)
    
    except Exception as e:
        logger.exception("Fatal error during validation:")
        sys.exit(1)


if __name__ == "__main__":
    main()