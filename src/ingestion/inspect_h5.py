import h5py
import yaml
import logging
import os
from datetime import datetime
from collections import defaultdict
import json

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = f"{log_dir}/inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger("H5_Inspector")


class H5Inspector:
    """Comprehensive HDF5 file inspector with logging"""
    
    def __init__(self, config_path="params.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.file_path = self.config['data_ingestion']['local_path']
        self.stats = {
            'total_groups': 0,
            'total_datasets': 0,
            'layouts': {},
            'dtypes_found': set(),
            'shapes_found': set(),
            'metadata': {}
        }
    
    def inspect_all(self):
        """Run all inspection tasks"""
        logger.info("="*70)
        logger.info("STARTING HDF5 INSPECTION")
        logger.info("="*70)
        logger.info(f"File: {self.file_path}")
        logger.info(f"File size: {os.path.getsize(self.file_path) / 1e6:.2f} MB")
        logger.info("")
        
        with h5py.File(self.file_path, 'r') as f:
            # 1. List all keys
            self._list_keys(f)
            
            # 2. Inspect shapes and dtypes
            self._inspect_shapes_dtypes(f)
            
            # 3. Read attributes/metadata
            self._read_attributes(f)
            
            # 4. Identify dimensions
            self._identify_dimensions(f)
            
            # 5. Take sample slices
            self._sample_slices(f)
            
            # 6. Generate summary
            self._generate_summary()
        
        logger.info("="*70)
        logger.info("INSPECTION COMPLETE")
        logger.info(f"Log saved to: {log_file}")
        logger.info("="*70)
    
    def _list_keys(self, f):
        """Task 1: List all HDF5 keys"""
        logger.info("\n" + "="*70)
        logger.info("TASK 1: LISTING ALL HDF5 KEYS")
        logger.info("="*70)
        
        all_keys = []
        
        def collect_keys(name, obj):
            all_keys.append(name)
            if isinstance(obj, h5py.Group):
                self.stats['total_groups'] += 1
            elif isinstance(obj, h5py.Dataset):
                self.stats['total_datasets'] += 1
        
        f.visititems(collect_keys)
        
        logger.info(f"Total keys found: {len(all_keys)}")
        logger.info(f"  - Groups: {self.stats['total_groups']}")
        logger.info(f"  - Datasets: {self.stats['total_datasets']}")
        
        # Log first 20 keys as sample
        logger.info(f"\nFirst 20 keys:")
        for i, key in enumerate(all_keys[:20], 1):
            logger.info(f"  {i:3d}. {key}")
        
        if len(all_keys) > 20:
            logger.info(f"  ... and {len(all_keys) - 20} more keys")
    
    def _inspect_shapes_dtypes(self, f):
        """Task 2: Inspect all shapes and dtypes"""
        logger.info("\n" + "="*70)
        logger.info("TASK 2: INSPECTING SHAPES AND DTYPES")
        logger.info("="*70)
        
        shape_dtype_map = defaultdict(list)
        
        def collect_shapes_dtypes(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = str(obj.dtype)
                
                self.stats['dtypes_found'].add(dtype)
                self.stats['shapes_found'].add(str(shape))
                
                key = f"Shape: {shape}, Dtype: {dtype}"
                shape_dtype_map[key].append(name)
        
        f.visititems(collect_shapes_dtypes)
        
        logger.info(f"\nUnique data types found: {len(self.stats['dtypes_found'])}")
        for dtype in sorted(self.stats['dtypes_found']):
            logger.info(f"  - {dtype}")
        
        logger.info(f"\nShape/Dtype combinations found:")
        for combo, datasets in sorted(shape_dtype_map.items()):
            logger.info(f"\n  {combo}")
            logger.info(f"    Count: {len(datasets)} datasets")
            # Log first 3 examples
            for example in datasets[:3]:
                logger.info(f"      Example: {example}")
            if len(datasets) > 3:
                logger.info(f"      ... and {len(datasets) - 3} more")
    
    def _read_attributes(self, f):
        """Task 3: Read all attributes/metadata"""
        logger.info("\n" + "="*70)
        logger.info("TASK 3: READING ATTRIBUTES/METADATA")
        logger.info("="*70)
        
        # File-level attributes
        logger.info("\nFile-level attributes:")
        if len(f.attrs) > 0:
            for key, value in f.attrs.items():
                logger.info(f"  {key}: {value}")
                self.stats['metadata'][key] = str(value)
        else:
            logger.info("  No file-level attributes found")
        
        # Sample group/dataset attributes
        logger.info("\nSample group/dataset attributes:")
        
        attr_count = 0
        
        def collect_attributes(name, obj):
            nonlocal attr_count
            if len(obj.attrs) > 0 and attr_count < 10:
                logger.info(f"\n  {name}:")
                for key, value in obj.attrs.items():
                    logger.info(f"    {key}: {value}")
                attr_count += 1
        
        f.visititems(collect_attributes)
        
        if attr_count == 0:
            logger.info("  No attributes found on groups/datasets")
    
    def _identify_dimensions(self, f):
        """Task 4: Identify time/entity dimensions"""
        logger.info("\n" + "="*70)
        logger.info("TASK 4: IDENTIFYING TIME/ENTITY DIMENSIONS")
        logger.info("="*70)
        
        # Analyze structure
        layouts = list(f.keys())
        logger.info(f"\nTotal Layouts (farm configurations): {len(layouts)}")
        
        # Analyze first layout in detail
        if layouts:
            first_layout = layouts[0]
            layout_obj = f[first_layout]
            
            logger.info(f"\nAnalyzing '{first_layout}':")
            
            if 'Scenarios' in layout_obj:
                scenarios = list(layout_obj['Scenarios'].keys())
                logger.info(f"  Total Scenarios (weather conditions): {len(scenarios)}")
                
                # Analyze first scenario
                first_scenario = layout_obj['Scenarios'][scenarios[0]]
                logger.info(f"\n  Analyzing '{scenarios[0]}':")
                
                for key in first_scenario.keys():
                    dataset = first_scenario[key]
                    shape = dataset.shape
                    
                    if shape == ():
                        dim_type = "Scalar (same for all turbines)"
                    else:
                        dim_type = f"Array (per-turbine, {shape[0]} turbines)"
                    
                    logger.info(f"    {key}: {dim_type}")
                
                # Store stats
                self.stats['layouts'][first_layout] = {
                    'num_scenarios': len(scenarios),
                    'num_turbines': first_scenario['Turbine Power'].shape[0]
                }
        
        # Dimension summary
        logger.info(f"\n📊 DIMENSION HIERARCHY:")
        logger.info(f"  Layout (farm configuration)")
        logger.info(f"    └── Scenario (weather condition)")
        logger.info(f"          ├── Scalar features (wind speed, direction, turbulence)")
        logger.info(f"          └── Turbine features (power, wind speed, yaw angles)")
    
    def _sample_slices(self, f):
        """Task 5: Take small slices to confirm structure"""
        logger.info("\n" + "="*70)
        logger.info("TASK 5: SAMPLING DATA SLICES")
        logger.info("="*70)
        
        layouts = list(f.keys())
        
        if not layouts:
            logger.warning("No layouts found in file")
            return
        
        # Sample from first layout
        first_layout = layouts[0]
        layout = f[first_layout]
        
        if 'Scenarios' not in layout:
            logger.warning(f"No 'Scenarios' group in {first_layout}")
            return
        
        scenarios = list(layout['Scenarios'].keys())
        first_scenario = layout['Scenarios'][scenarios[0]]
        
        logger.info(f"\nSampling from: {first_layout}/Scenarios/{scenarios[0]}")
        logger.info("")
        
        # Sample scalar values
        logger.info("Scalar Values:")
        logger.info(f"  Wind Speed: {first_scenario['Wind Speed'][()]:.2f} m/s")
        logger.info(f"  Wind Direction: {first_scenario['Wind Direction'][()]:.2f} degrees")
        logger.info(f"  Turbulence Intensity: {first_scenario['Turbulence Intensity'][()]:.6f}")
        
        # Sample array values
        logger.info("\nArray Values (first 5 turbines):")
        logger.info(f"  Turbine Power: {first_scenario['Turbine Power'][:5]}")
        logger.info(f"  Turbine Wind Speed: {first_scenario['Turbine Wind Speed'][:5]}")
        logger.info(f"  Yaw Angles: {first_scenario['Yaw Angles'][:5]}")
        
        # Sample from multiple layouts if available
        if len(layouts) > 1:
            logger.info(f"\n\nSampling from another layout: {layouts[1]}")
            layout2 = f[layouts[1]]
            scenarios2 = list(layout2['Scenarios'].keys())
            scenario2 = layout2['Scenarios'][scenarios2[0]]
            
            logger.info(f"  Wind Speed: {scenario2['Wind Speed'][()]:.2f} m/s")
            logger.info(f"  Number of turbines: {scenario2['Turbine Power'].shape[0]}")
            logger.info(f"  First 3 powers: {scenario2['Turbine Power'][:3]}")
    
    def _generate_summary(self):
        """Generate final summary statistics"""
        logger.info("\n" + "="*70)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*70)
        
        logger.info(f"\nStructure:")
        logger.info(f"  Total Groups: {self.stats['total_groups']}")
        logger.info(f"  Total Datasets: {self.stats['total_datasets']}")
        
        logger.info(f"\nData Types Found: {len(self.stats['dtypes_found'])}")
        for dtype in sorted(self.stats['dtypes_found']):
            logger.info(f"  - {dtype}")
        
        logger.info(f"\nUnique Shapes Found: {len(self.stats['shapes_found'])}")
        for shape in sorted(self.stats['shapes_found']):
            logger.info(f"  - {shape}")
        
        # Save summary as JSON
        summary_file = f"{log_dir}/inspection_summary.json"
        
        summary_dict = {
            'file_path': self.file_path,
            'file_size_mb': os.path.getsize(self.file_path) / 1e6,
            'total_groups': self.stats['total_groups'],
            'total_datasets': self.stats['total_datasets'],
            'dtypes': list(self.stats['dtypes_found']),
            'shapes': list(self.stats['shapes_found']),
            'metadata': self.stats['metadata'],
            'sample_layout_stats': self.stats['layouts']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logger.info(f"\n💾 Summary saved to: {summary_file}")


def main():
    inspector = H5Inspector()
    inspector.inspect_all()


if __name__ == "__main__":
    main()