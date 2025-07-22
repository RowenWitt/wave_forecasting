# diagnose_data_issues.py
"""
Diagnose data loading issues - find the actual variable names and check for null/nan values
"""

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_actual_data_structure(file_path: str):
    """Thoroughly inspect the actual data structure"""
    
    print(f"🔍 DEEP INSPECTION: {Path(file_path).name}")
    print("=" * 80)
    
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"📊 Dataset Overview:")
            print(f"   Dimensions: {dict(ds.dims)}")
            print(f"   Coordinates: {list(ds.coords.keys())}")
            print(f"   Total variables: {len(ds.data_vars)}")
            
            # Show ALL variables with their actual names and statistics
            print(f"\n📋 ALL VARIABLES WITH STATISTICS:")
            print(f"{'Variable Name':<35} {'Dims':<25} {'Shape':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'NaN%'}")
            print("-" * 120)
            
            for var_name in sorted(ds.data_vars.keys()):
                var_data = ds[var_name]
                values = var_data.values
                
                # Calculate statistics
                finite_mask = np.isfinite(values)
                if np.any(finite_mask):
                    min_val = np.min(values[finite_mask])
                    max_val = np.max(values[finite_mask])
                    mean_val = np.mean(values[finite_mask])
                else:
                    min_val = max_val = mean_val = np.nan
                
                nan_percent = (np.isnan(values).sum() / values.size) * 100
                
                print(f"{var_name:<35} {str(var_data.dims):<25} {str(var_data.shape):<20} "
                      f"{min_val:<12.3f} {max_val:<12.3f} {mean_val:<12.3f} {nan_percent:<8.1f}%")
            
            # Look for wave-related variables specifically
            print(f"\n🌊 WAVE VARIABLE DETECTION:")
            wave_keywords = ['swh', 'wave', 'height', 'direction', 'period', 'significant']
            
            found_wave_vars = []
            for var_name in ds.data_vars.keys():
                var_lower = var_name.lower()
                for keyword in wave_keywords:
                    if keyword in var_lower:
                        found_wave_vars.append(var_name)
                        break
            
            if found_wave_vars:
                print(f"   Found potential wave variables:")
                for var in found_wave_vars:
                    print(f"   ✅ {var}")
            else:
                print(f"   ❌ No obvious wave variables found!")
                print(f"   🔍 Looking for similar patterns...")
                
                # Look for any numeric variables that might be waves
                numeric_vars = []
                for var_name in ds.data_vars.keys():
                    var_data = ds[var_name]
                    if len(var_data.dims) == 3:  # time, lat, lon
                        values = var_data.values
                        if np.issubdtype(values.dtype, np.number):
                            numeric_vars.append(var_name)
                
                print(f"   📊 All 3D numeric variables:")
                for var in numeric_vars:
                    print(f"      {var}")
            
            # Check specific coordinate structure
            print(f"\n🗺️  COORDINATE ANALYSIS:")
            for coord_name in ds.coords:
                coord_data = ds.coords[coord_name]
                print(f"   {coord_name}: {coord_data.shape} - {coord_data.values[:5]}...")
                
                if 'time' in coord_name.lower():
                    print(f"      Time range: {coord_data.values[0]} to {coord_data.values[-1]}")
            
            # Sample a few variables to check actual data quality
            print(f"\n🔬 DATA QUALITY SAMPLE:")
            sample_vars = list(ds.data_vars.keys())[:5]
            
            for var_name in sample_vars:
                var_data = ds[var_name]
                values = var_data.values
                
                # Take a small sample
                if len(values.shape) >= 3:
                    sample = values[0, :5, :5]  # First time, 5x5 spatial
                else:
                    sample = values.flat[:25]  # First 25 values
                
                print(f"   {var_name}:")
                print(f"      Sample values: {sample.flatten()[:10]}")
                print(f"      Unique values: {len(np.unique(values[np.isfinite(values)]))}")
                
                # Check if all zeros
                if np.allclose(values[np.isfinite(values)], 0, atol=1e-10):
                    print(f"      ⚠️  ALL VALUES ARE ZERO!")
                
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")
        import traceback
        traceback.print_exc()

def find_wave_variables_by_pattern(file_path: str):
    """Try to identify wave variables by examining data patterns"""
    
    print(f"\n🕵️ WAVE VARIABLE DETECTIVE")
    print("=" * 50)
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Analyze each variable to guess what it might be
            candidates = {
                'swh': [],
                'mwd': [],
                'mwp': []
            }
            
            for var_name in ds.data_vars.keys():
                var_data = ds[var_name]
                
                # Only consider 3D variables (time, lat, lon)
                if len(var_data.dims) != 3:
                    continue
                
                values = var_data.values
                finite_values = values[np.isfinite(values)]
                
                if len(finite_values) == 0:
                    continue
                
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
                mean_val = np.mean(finite_values)
                
                # SWH detection (significant wave height: typically 0-20m)
                if 0 <= min_val < 0.5 and 0.5 < max_val < 25 and 0.5 < mean_val < 10:
                    candidates['swh'].append((var_name, min_val, max_val, mean_val))
                
                # MWD detection (mean wave direction: typically 0-360 degrees)
                if 0 <= min_val < 30 and 300 < max_val <= 360 and 50 < mean_val < 300:
                    candidates['mwd'].append((var_name, min_val, max_val, mean_val))
                
                # MWP detection (mean wave period: typically 2-20 seconds)
                if 1 < min_val < 5 and 5 < max_val < 25 and 3 < mean_val < 15:
                    candidates['mwp'].append((var_name, min_val, max_val, mean_val))
            
            print(f"🎯 WAVE VARIABLE CANDIDATES:")
            
            for wave_type, candidate_list in candidates.items():
                print(f"\n   {wave_type.upper()} candidates:")
                if candidate_list:
                    for var_name, min_val, max_val, mean_val in candidate_list:
                        print(f"      ✅ {var_name}: range [{min_val:.2f}, {max_val:.2f}], mean {mean_val:.2f}")
                else:
                    print(f"      ❌ No candidates found")
            
            # If no clear candidates, show variables with reasonable ranges
            print(f"\n📊 ALL VARIABLES WITH REASONABLE RANGES:")
            for var_name in ds.data_vars.keys():
                var_data = ds[var_name]
                
                if len(var_data.dims) != 3:
                    continue
                
                values = var_data.values
                finite_values = values[np.isfinite(values)]
                
                if len(finite_values) == 0:
                    continue
                
                min_val = np.min(finite_values)
                max_val = np.max(finite_values)
                mean_val = np.mean(finite_values)
                std_val = np.std(finite_values)
                
                # Show any variable with non-zero variance
                if std_val > 1e-6:
                    print(f"   {var_name:30} range: [{min_val:8.3f}, {max_val:8.3f}], mean: {mean_val:8.3f}, std: {std_val:8.3f}")
            
    except Exception as e:
        print(f"❌ Error in wave detection: {e}")

def create_corrected_variable_mapping(file_path: str):
    """Create a corrected variable mapping based on actual file contents"""
    
    print(f"\n🔧 CREATING CORRECTED VARIABLE MAPPING")
    print("=" * 50)
    
    try:
        with xr.open_dataset(file_path) as ds:
            available_vars = list(ds.data_vars.keys())
            
            print(f"Available variables: {available_vars}")
            
            # Create a mapping based on what's actually available
            corrected_mapping = {}
            
            # Standard atmospheric variables (exact name matching first)
            standard_atm = {
                'u10': ['u10', '10m_u_component_of_wind'],
                'v10': ['v10', '10m_v_component_of_wind'],
                'slp': ['slp', 'msl', 'mean_sea_level_pressure'],
                'sst': ['sst', 'sea_surface_temperature'],
                'precip': ['precip', 'tp', 'total_precipitation']
            }
            
            # Pressure level variables
            pressure_vars = {
                'u850': ['u850'], 'v850': ['v850'], 'z500': ['z500'],
                'u500': ['u500'], 'v500': ['v500'], 'z850': ['z850']
            }
            
            # Try to find each variable
            found_vars = []
            
            for target_var, possible_names in {**standard_atm, **pressure_vars}.items():
                found = False
                for possible_name in possible_names:
                    if possible_name in available_vars:
                        corrected_mapping[target_var] = possible_name
                        found_vars.append(f"✅ {target_var} -> {possible_name}")
                        found = True
                        break
                
                if not found:
                    found_vars.append(f"❌ {target_var} -> NOT FOUND")
            
            # Special handling for wave variables - be more flexible
            wave_vars = []
            for var in available_vars:
                var_lower = var.lower()
                
                # SWH patterns
                if any(pattern in var_lower for pattern in ['swh', 'significant', 'wave_height', 'hs']):
                    wave_vars.append(('swh', var))
                
                # MWD patterns  
                elif any(pattern in var_lower for pattern in ['mwd', 'direction', 'wave_dir', 'dir']):
                    wave_vars.append(('mwd', var))
                
                # MWP patterns
                elif any(pattern in var_lower for pattern in ['mwp', 'period', 'wave_period', 'tp']):
                    # Exclude total_precipitation (tp) 
                    if 'precip' not in var_lower and 'rain' not in var_lower:
                        wave_vars.append(('mwp', var))
            
            print(f"\n📋 CORRECTED MAPPING:")
            for var_info in found_vars:
                print(f"   {var_info}")
            
            print(f"\n🌊 WAVE VARIABLES:")
            for wave_type, var_name in wave_vars:
                print(f"   ✅ {wave_type} -> {var_name}")
                corrected_mapping[wave_type] = var_name
            
            # Generate Python code for the corrected mapping
            print(f"\n🐍 CORRECTED PYTHON MAPPING:")
            print("CORRECTED_VARIABLE_MAPPING = {")
            for target_var, actual_var in corrected_mapping.items():
                print(f"    '{target_var}': ['{actual_var}'],")
            print("}")
            
            return corrected_mapping
            
    except Exception as e:
        print(f"❌ Error creating mapping: {e}")
        return {}

def test_corrected_loading(file_path: str, corrected_mapping: dict):
    """Test loading data with the corrected mapping"""
    
    print(f"\n🧪 TESTING CORRECTED LOADING")
    print("=" * 50)
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            print(f"Testing extraction with corrected mapping...")
            
            extracted_vars = []
            var_names = []
            
            for target_var, actual_var in corrected_mapping.items():
                if actual_var in ds.data_vars:
                    var_data = ds[actual_var]
                    
                    if len(var_data.dims) == 3:  # time, lat, lon
                        values = var_data.values
                        
                        print(f"   ✅ {target_var} ({actual_var}): shape {values.shape}")
                        print(f"      Range: [{np.nanmin(values):.3f}, {np.nanmax(values):.3f}]")
                        print(f"      Mean: {np.nanmean(values):.3f}, NaN%: {(np.isnan(values).sum()/values.size)*100:.1f}%")
                        
                        extracted_vars.append(values)
                        var_names.append(target_var)
                    else:
                        print(f"   ⚠️  {target_var} ({actual_var}): wrong dimensions {var_data.dims}")
                else:
                    print(f"   ❌ {target_var}: {actual_var} not found")
            
            if len(extracted_vars) >= 3:
                print(f"\n✅ Successfully extracted {len(extracted_vars)} variables")
                
                # Check wave variables specifically
                wave_indices = {}
                for i, var_name in enumerate(var_names):
                    if var_name in ['swh', 'mwd', 'mwp']:
                        wave_indices[var_name] = i
                
                print(f"   Wave variable indices: {wave_indices}")
                
                if len(wave_indices) == 3:
                    print(f"   ✅ All 3 wave variables found!")
                    
                    # Show sample wave data
                    for wave_var, idx in wave_indices.items():
                        sample_data = extracted_vars[idx][0, 10:15, 10:15]  # Small sample
                        print(f"   {wave_var} sample: {sample_data.flatten()[:5]}")
                else:
                    print(f"   ⚠️  Only found {len(wave_indices)}/3 wave variables")
            else:
                print(f"   ❌ Only extracted {len(extracted_vars)} variables")
            
    except Exception as e:
        print(f"❌ Error testing corrected loading: {e}")

def main():
    """Main diagnostic function"""
    
    print("🩺 DATA LOADING DIAGNOSIS")
    print("=" * 80)
    
    # Find test file
    data_dir = Path("data/era5_global")
    
    test_files = []
    # for pattern in ["era5_smart_joined_*.nc", "era5_joined_*.nc"]:
    for pattern in ["era5_single_level_201909.nc", ]:
        test_files.extend(data_dir.glob(pattern))
    
    if not test_files:
        print("❌ No test files found!")
        return
    
    test_file = str(test_files[0])
    print(f"🎯 Diagnosing: {Path(test_file).name}")
    
    # Run diagnostics
    inspect_actual_data_structure(test_file)
    find_wave_variables_by_pattern(test_file)
    corrected_mapping = create_corrected_variable_mapping(test_file)
    
    if corrected_mapping:
        test_corrected_loading(test_file, corrected_mapping)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Use the corrected mapping above in your model")
    print(f"   2. Update FULL_VARIABLE_MAPPING with actual variable names")
    print(f"   3. Re-run training with corrected variable names")

if __name__ == "__main__":
    main()