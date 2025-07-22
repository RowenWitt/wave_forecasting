# fixed_spatial_temporal_analysis.py
"""
Fixed analysis with spatial bounds checking and proper temporal handling
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
import re
warnings.filterwarnings('ignore')

def analyze_spatial_bounds(file_path: str, file_type: str):
    """Analyze spatial bounds and coverage of variables"""
    
    print(f"\n🗺️  SPATIAL BOUNDS ANALYSIS: {file_type}")
    print(f"   File: {Path(file_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Find spatial coordinates
            lat_coord = lon_coord = None
            for coord in ds.coords:
                if 'lat' in coord.lower():
                    lat_coord = coord
                elif 'lon' in coord.lower():
                    lon_coord = coord
            
            if not lat_coord or not lon_coord:
                print(f"   ❌ Missing spatial coordinates")
                return None
            
            lat_values = ds[lat_coord].values
            lon_values = ds[lon_coord].values
            
            print(f"   📍 Spatial coordinates:")
            print(f"      Latitude ({lat_coord}): [{lat_values.min():.2f}, {lat_values.max():.2f}]°")
            print(f"      Longitude ({lon_coord}): [{lon_values.min():.2f}, {lon_values.max():.2f}]°")
            print(f"      Grid size: {len(lat_values)} × {len(lon_values)} = {len(lat_values) * len(lon_values):,} points")
            
            # Analyze each variable's spatial coverage
            print(f"\n   📊 Variable Spatial Coverage:")
            
            for var_name in list(ds.data_vars.keys())[:10]:  # First 10 variables
                var_data = ds[var_name]
                
                if len(var_data.dims) >= 2:  # Has spatial dimensions
                    # Take first time slice if 3D
                    if len(var_data.dims) == 3:
                        # Find first time with data
                        time_dim = var_data.dims[0]
                        spatial_slice = None
                        for t in range(min(10, var_data.shape[0])):  # Check first 10 timesteps
                            test_slice = var_data.isel({time_dim: t})
                            if np.any(np.isfinite(test_slice.values)):
                                spatial_slice = test_slice
                                break
                        
                        if spatial_slice is None:
                            print(f"      {var_name:25} No finite data found in first 10 timesteps")
                            continue
                    else:
                        spatial_slice = var_data
                    
                    values = spatial_slice.values
                    finite_mask = np.isfinite(values)
                    
                    if np.any(finite_mask):
                        # Find bounding box of finite data
                        finite_indices = np.where(finite_mask)
                        
                        if len(finite_indices) == 2:  # 2D data
                            lat_indices, lon_indices = finite_indices
                            
                            min_lat_idx = lat_indices.min()
                            max_lat_idx = lat_indices.max()
                            min_lon_idx = lon_indices.min()
                            max_lon_idx = lon_indices.max()
                            
                            min_lat = lat_values[min_lat_idx]
                            max_lat = lat_values[max_lat_idx]
                            min_lon = lon_values[min_lon_idx]
                            max_lon = lon_values[max_lon_idx]
                            
                            coverage_percent = (finite_mask.sum() / values.size) * 100
                            
                            print(f"      {var_name:25} Coverage: {coverage_percent:5.1f}% | "
                                  f"Lat: [{min_lat:6.1f}, {max_lat:6.1f}]° | "
                                  f"Lon: [{min_lon:6.1f}, {max_lon:6.1f}]°")
                            
                            # Check if it's ocean-only (common pattern for wave data)
                            if coverage_percent < 50:
                                print(f"      {' '*25}        ↳ Likely ocean-only variable")
                    else:
                        print(f"      {var_name:25} No finite data")
                else:
                    print(f"      {var_name:25} Not spatial data")
            
            if len(ds.data_vars) > 10:
                print(f"      ... and {len(ds.data_vars) - 10} more variables")
            
            return {
                'lat_coord': lat_coord,
                'lon_coord': lon_coord,
                'lat_bounds': (lat_values.min(), lat_values.max()),
                'lon_bounds': (lon_values.min(), lon_values.max()),
                'grid_shape': (len(lat_values), len(lon_values))
            }
            
    except Exception as e:
        print(f"   ❌ Error in spatial analysis: {e}")
        return None

def analyze_temporal_structure_fixed(file_path: str, file_type: str):
    """Fixed temporal structure analysis"""
    
    print(f"\n⏰ TEMPORAL ANALYSIS: {file_type}")
    print(f"   File: {Path(file_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            # Find time coordinate
            time_coord = None
            for coord in ds.coords:
                if 'time' in coord.lower():
                    time_coord = coord
                    break
            
            if not time_coord:
                print(f"   ❌ No time coordinate found")
                return None
            
            time_values = ds[time_coord].values
            
            # Convert to pandas datetime for easier handling
            time_pandas = pd.to_datetime(time_values)
            
            print(f"   📊 Time coordinate: {time_coord}")
            print(f"   📅 Time range: {time_pandas[0]} to {time_pandas[-1]}")
            print(f"   📈 Total timesteps: {len(time_values)}")
            
            # Calculate time differences - FIXED VERSION
            if len(time_pandas) > 1:
                time_diffs = time_pandas[1:] - time_pandas[:-1]
                
                # Convert to hours
                time_diffs_hours = time_diffs.total_seconds() / 3600
                unique_diffs_hours = np.unique(time_diffs_hours)
                
                print(f"   ⏱️  Time intervals:")
                for diff_hours in unique_diffs_hours:
                    count = np.sum(time_diffs_hours == diff_hours)
                    print(f"      {diff_hours:5.1f} hours: {count:4d} intervals")
                
                # Determine dominant interval
                most_common_diff = unique_diffs_hours[np.argmax([np.sum(time_diffs_hours == diff) for diff in unique_diffs_hours])]
                print(f"   🎯 Dominant interval: {most_common_diff:.1f} hours")
                
                return {
                    'time_coord': time_coord,
                    'time_values': time_values,
                    'time_pandas': time_pandas,
                    'dominant_interval_hours': most_common_diff,
                    'total_steps': len(time_values),
                    'time_diffs_hours': time_diffs_hours
                }
            else:
                print(f"   ⚠️  Only one timestep found")
                return None
            
    except Exception as e:
        print(f"   ❌ Error analyzing temporal structure: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_ocean_coverage_detailed(file_path: str):
    """Detailed analysis of ocean coverage patterns"""
    
    print(f"\n🌊 DETAILED OCEAN COVERAGE ANALYSIS")
    print(f"   File: {Path(file_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Find wave variables
            wave_vars = {}
            for var_name in ds.data_vars:
                var_lower = var_name.lower()
                if any(pattern in var_lower for pattern in ['swh', 'significant', 'wave_height']):
                    wave_vars['swh'] = var_name
                elif any(pattern in var_lower for pattern in ['mwd', 'direction', 'wave_dir']):
                    wave_vars['mwd'] = var_name
                elif any(pattern in var_lower for pattern in ['mwp', 'period', 'wave_period']):
                    if 'precip' not in var_lower:
                        wave_vars['mwp'] = var_name
            
            if not wave_vars:
                print(f"   ❌ No wave variables found")
                return
            
            # Get spatial coordinates
            lat_coord = lon_coord = None
            for coord in ds.coords:
                if 'lat' in coord.lower():
                    lat_coord = coord
                elif 'lon' in coord.lower():
                    lon_coord = coord
            
            lat_values = ds[lat_coord].values
            lon_values = ds[lon_coord].values
            
            # Analyze ocean mask using SWH (if available)
            if 'swh' in wave_vars:
                swh_var = ds[wave_vars['swh']]
                
                # Find a good timestep with data
                ocean_mask = None
                for t in range(min(20, swh_var.shape[0])):
                    time_slice = swh_var.isel({swh_var.dims[0]: t})
                    if np.any(np.isfinite(time_slice.values)):
                        ocean_mask = np.isfinite(time_slice.values)
                        break
                
                if ocean_mask is not None:
                    # Analyze ocean distribution
                    ocean_points = np.where(ocean_mask)
                    
                    if len(ocean_points[0]) > 0:
                        # Find geographic bounds of ocean data
                        lat_indices = ocean_points[0]
                        lon_indices = ocean_points[1]
                        
                        ocean_lats = lat_values[lat_indices]
                        ocean_lons = lon_values[lon_indices]
                        
                        print(f"   🌊 Ocean Data Geographic Bounds:")
                        print(f"      Latitude range: [{ocean_lats.min():.2f}°, {ocean_lats.max():.2f}°]")
                        print(f"      Longitude range: [{ocean_lons.min():.2f}°, {ocean_lons.max():.2f}°]")
                        print(f"      Ocean points: {len(ocean_lats):,}")
                        print(f"      Total grid points: {lat_values.size * lon_values.size:,}")
                        print(f"      Ocean coverage: {(len(ocean_lats) / (lat_values.size * lon_values.size)) * 100:.1f}%")
                        
                        # Check for specific ocean regions
                        regions = {
                            'North Atlantic': (lat_values >= 30) & (lat_values <= 70) & (lon_values >= -80) & (lon_values <= 10),
                            'North Pacific': (lat_values >= 30) & (lat_values <= 70) & (lon_values >= 120) | (lon_values <= -120),
                            'Southern Ocean': lat_values <= -40,
                            'Tropical': (lat_values >= -30) & (lat_values <= 30),
                            'Arctic': lat_values >= 70
                        }
                        
                        print(f"\n   🗺️  Regional Ocean Coverage:")
                        
                        for region_name, region_mask in regions.items():
                            # Create 2D mask
                            lat_2d, lon_2d = np.meshgrid(lat_values, lon_values, indexing='ij')
                            region_2d = region_mask
                            
                            # Count ocean points in this region
                            region_ocean = ocean_mask & region_2d
                            region_ocean_count = region_ocean.sum()
                            region_total = region_2d.sum()
                            
                            if region_total > 0:
                                region_coverage = (region_ocean_count / region_total) * 100
                                print(f"      {region_name:15}: {region_ocean_count:6,} ocean points / {region_total:6,} total ({region_coverage:5.1f}%)")
                        
                        # Check temporal consistency
                        print(f"\n   📅 Temporal Coverage Analysis:")
                        
                        time_steps_with_data = []
                        for t in range(min(50, swh_var.shape[0])):  # Check first 50 timesteps
                            time_slice = swh_var.isel({swh_var.dims[0]: t})
                            valid_ocean_points = np.isfinite(time_slice.values).sum()
                            time_steps_with_data.append(valid_ocean_points)
                        
                        time_steps_with_data = np.array(time_steps_with_data)
                        
                        print(f"      Timesteps analyzed: {len(time_steps_with_data)}")
                        print(f"      Min ocean points per timestep: {time_steps_with_data.min():,}")
                        print(f"      Max ocean points per timestep: {time_steps_with_data.max():,}")
                        print(f"      Mean ocean points per timestep: {time_steps_with_data.mean():.0f}")
                        
                        # Consistency check
                        consistent_timesteps = np.sum(time_steps_with_data > 1000)  # At least 1000 ocean points
                        consistency_percent = (consistent_timesteps / len(time_steps_with_data)) * 100
                        
                        print(f"      Timesteps with >1000 ocean points: {consistent_timesteps}/{len(time_steps_with_data)} ({consistency_percent:.1f}%)")
                        
                        if consistency_percent > 50:
                            print(f"   ✅ Ocean data appears CONSISTENT across time")
                        else:
                            print(f"   ⚠️  Ocean data has temporal inconsistencies")
                        
                        return {
                            'ocean_bounds': {
                                'lat_min': ocean_lats.min(),
                                'lat_max': ocean_lats.max(),
                                'lon_min': ocean_lons.min(),
                                'lon_max': ocean_lons.max()
                            },
                            'ocean_coverage_percent': (len(ocean_lats) / (lat_values.size * lon_values.size)) * 100,
                            'temporal_consistency_percent': consistency_percent,
                            'usable': consistency_percent > 50
                        }
                
    except Exception as e:
        print(f"   ❌ Error in ocean coverage analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def assess_data_quality_for_training(single_file: str, pressure_file: str):
    """Comprehensive assessment of data quality for training"""
    
    print(f"\n🎯 COMPREHENSIVE DATA QUALITY ASSESSMENT")
    print(f"=" * 60)
    
    # Analyze both files
    print(f"📊 SINGLE-LEVEL DATA:")
    single_spatial = analyze_spatial_bounds(single_file, "Single-level")
    single_temporal = analyze_temporal_structure_fixed(single_file, "Single-level")
    ocean_analysis = analyze_ocean_coverage_detailed(single_file)
    
    print(f"\n📊 PRESSURE-LEVEL DATA:")
    pressure_spatial = analyze_spatial_bounds(pressure_file, "Pressure-level")
    pressure_temporal = analyze_temporal_structure_fixed(pressure_file, "Pressure-level")
    
    # Overall assessment
    print(f"\n🎯 TRAINING SUITABILITY ASSESSMENT:")
    
    issues = []
    recommendations = []
    
    # Check spatial compatibility
    if single_spatial and pressure_spatial:
        if single_spatial['grid_shape'] == pressure_spatial['grid_shape']:
            print(f"   ✅ Spatial grids match: {single_spatial['grid_shape']}")
        else:
            print(f"   ⚠️  Spatial grid mismatch: {single_spatial['grid_shape']} vs {pressure_spatial['grid_shape']}")
            issues.append("Spatial grid mismatch")
    
    # Check temporal alignment
    if single_temporal and pressure_temporal:
        single_interval = single_temporal['dominant_interval_hours']
        pressure_interval = pressure_temporal['dominant_interval_hours']
        
        print(f"   📅 Temporal intervals: {single_interval:.1f}h (single) vs {pressure_interval:.1f}h (pressure)")
        
        if abs(single_interval - pressure_interval) < 0.1:
            print(f"   ✅ Temporal intervals match")
        else:
            print(f"   ⚠️  Temporal interval mismatch - alignment needed")
            recommendations.append("Use temporal alignment")
            
            # Calculate alignment ratio
            ratio = max(single_interval, pressure_interval) / min(single_interval, pressure_interval)
            if ratio <= 3:
                print(f"   ✅ Alignment ratio {ratio:.1f}x is reasonable")
            else:
                print(f"   ❌ Alignment ratio {ratio:.1f}x is too high")
                issues.append("Temporal resolution too different")
    
    # Check ocean data quality
    if ocean_analysis:
        coverage = ocean_analysis['ocean_coverage_percent']
        consistency = ocean_analysis['temporal_consistency_percent']
        
        print(f"   🌊 Ocean coverage: {coverage:.1f}% of global grid")
        print(f"   📅 Temporal consistency: {consistency:.1f}% of timesteps")
        
        if coverage > 10:
            print(f"   ✅ Ocean coverage is reasonable")
        else:
            print(f"   ❌ Ocean coverage too low")
            issues.append("Insufficient ocean coverage")
        
        if consistency > 70:
            print(f"   ✅ Temporal consistency is good")
        elif consistency > 50:
            print(f"   ⚠️  Temporal consistency is marginal")
            recommendations.append("Monitor training stability")
        else:
            print(f"   ❌ Poor temporal consistency")
            issues.append("Inconsistent temporal coverage")
    
    # Final recommendation
    print(f"\n🏁 FINAL RECOMMENDATION:")
    
    if len(issues) == 0:
        print(f"   ✅ Data appears SUITABLE for training!")
        print(f"   🚀 Proceed with model training")
    elif len(issues) <= 2 and len(recommendations) > 0:
        print(f"   ⚠️  Data is USABLE but needs preprocessing:")
        for rec in recommendations:
            print(f"      • {rec}")
    else:
        print(f"   ❌ Data has SERIOUS issues:")
        for issue in issues:
            print(f"      • {issue}")
        print(f"   🔄 Consider re-downloading with different parameters")
    
    return {
        'issues': issues,
        'recommendations': recommendations,
        'usable': len(issues) <= 2
    }

def main():
    """Main analysis function"""
    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python fixed_spatial_temporal_analysis.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    print(f"🔍 COMPREHENSIVE ERA5 DATA ANALYSIS")
    print(f"=" * 80)
    print(f"📁 Directory: {directory}")
    
    # Find file pairs
    single_files = list(directory.glob("era5_single_level_*.nc"))
    
    if not single_files:
        print(f"❌ No single-level files found")
        sys.exit(1)
    
    # Analyze first pair
    single_file = single_files[0]
    time_match = re.search(r'era5_single_level_(\d{6})\.nc', single_file.name)
    
    if not time_match:
        print(f"❌ Could not parse time from filename")
        sys.exit(1)
    
    time_key = time_match.group(1)
    pressure_file = directory / f"era5_pressure_levels_{time_key}.nc"
    
    if not pressure_file.exists():
        print(f"❌ No matching pressure file found")
        sys.exit(1)
    
    print(f"🎯 Analyzing pair: {time_key}")
    
    # Run comprehensive assessment
    assessment = assess_data_quality_for_training(str(single_file), str(pressure_file))
    
    print(f"\n📋 SUMMARY:")
    if assessment['usable']:
        print(f"   ✅ Data is usable for training")
        if assessment['recommendations']:
            print(f"   📝 Apply recommended preprocessing steps")
    else:
        print(f"   ❌ Data quality issues prevent training")
        print(f"   🔄 Consider re-downloading ERA5 data")

if __name__ == "__main__":
    main()