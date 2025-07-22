#!/usr/bin/env python3
"""
Inspect fresh ocean data from CDS API
Check NaN rates and data quality for wave variables
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_ocean_data(file_path: str):
    """Comprehensive inspection of ocean data file"""
    
    print(f"🌊 OCEAN DATA INSPECTION")
    print(f"=" * 60)
    print(f"File: {file_path}")
    
    try:
        ds = xr.open_dataset(file_path)
        
        # Basic file info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   File size: {Path(file_path).stat().st_size / (1024**2):.1f} MB")
        print(f"   Dimensions: {dict(ds.dims)}")
        print(f"   Variables: {len(ds.data_vars)}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        
        # List all variables
        print(f"\n📋 ALL VARIABLES:")
        for i, var_name in enumerate(ds.data_vars.keys(), 1):
            var_info = ds[var_name]
            print(f"   {i:2d}. {var_name}")
            print(f"       Dimensions: {var_info.dims}")
            print(f"       Shape: {var_info.shape}")
            if hasattr(var_info, 'long_name'):
                print(f"       Description: {var_info.long_name}")
            if hasattr(var_info, 'units'):
                print(f"       Units: {var_info.units}")
        
        # Time info
        time_coords = [coord for coord in ds.coords if 'time' in coord.lower()]
        if time_coords:
            time_coord = time_coords[0]
            time_values = ds[time_coord].values
            print(f"\n⏰ TIME INFORMATION:")
            print(f"   Time coordinate: {time_coord}")
            print(f"   Time steps: {len(time_values)}")
            print(f"   Time range: {pd.to_datetime(time_values[0])} to {pd.to_datetime(time_values[-1])}")
            print(f"   Time frequency: {pd.to_datetime(time_values[1]) - pd.to_datetime(time_values[0])}")
        
        # Spatial info
        lat_coord = None
        lon_coord = None
        for coord in ds.coords:
            if coord.lower() in ['lat', 'latitude']:
                lat_coord = coord
            elif coord.lower() in ['lon', 'longitude']:
                lon_coord = coord
        
        if lat_coord and lon_coord:
            lats = ds[lat_coord].values
            lons = ds[lon_coord].values
            
            print(f"\n🌍 SPATIAL INFORMATION:")
            print(f"   Latitude: {lat_coord}")
            print(f"     Range: {lats.min():.3f}° to {lats.max():.3f}°")
            print(f"     Resolution: ~{abs(lats[1] - lats[0]):.3f}°")
            print(f"     Points: {len(lats)}")
            
            print(f"   Longitude: {lon_coord}")
            print(f"     Range: {lons.min():.3f}° to {lons.max():.3f}°")
            print(f"     Resolution: ~{abs(lons[1] - lons[0]):.3f}°")
            print(f"     Points: {len(lons)}")
            
            total_grid_points = len(lats) * len(lons)
            total_space_time_points = total_grid_points * len(time_values) if time_coords else total_grid_points
            
            print(f"   Total spatial grid points: {total_grid_points:,}")
            print(f"   Total space-time points: {total_space_time_points:,}")
            
            # Estimate ocean coverage
            expected_ocean_points = int(total_grid_points * 0.71)  # ~71% ocean
            expected_land_points = total_grid_points - expected_ocean_points
            
            print(f"\n🌊 EXPECTED OCEAN COVERAGE:")
            print(f"   Ocean points expected: ~{expected_ocean_points:,} ({71:.1f}%)")
            print(f"   Land points expected: ~{expected_land_points:,} ({29:.1f}%)")
            print(f"   Expected NaN for ocean vars: ~29%")
        
        # Detailed analysis of each variable
        print(f"\n🔍 DETAILED VARIABLE ANALYSIS:")
        
        for var_name in ds.data_vars:
            var_data = ds[var_name]
            values = var_data.values
            
            print(f"\n   📊 {var_name.upper()}:")
            print(f"      Shape: {values.shape}")
            print(f"      Total points: {values.size:,}")
            
            # Data quality metrics
            nan_count = np.isnan(values).sum()
            inf_count = np.isinf(values).sum()
            finite_count = np.isfinite(values).sum()
            zero_count = (values == 0).sum()
            
            nan_percent = (nan_count / values.size) * 100
            finite_percent = (finite_count / values.size) * 100
            
            print(f"      NaN values: {nan_count:,} ({nan_percent:.1f}%)")
            print(f"      Infinite values: {inf_count:,}")
            print(f"      Finite values: {finite_count:,} ({finite_percent:.1f}%)")
            print(f"      Zero values: {zero_count:,}")
            
            # Statistical analysis of finite values
            if finite_count > 0:
                finite_vals = values[np.isfinite(values)]
                
                print(f"      📈 STATISTICS (finite values only):")
                print(f"         Min: {np.min(finite_vals):.6f}")
                print(f"         Max: {np.max(finite_vals):.6f}")
                print(f"         Mean: {np.mean(finite_vals):.6f}")
                print(f"         Std: {np.std(finite_vals):.6f}")
                print(f"         Median: {np.median(finite_vals):.6f}")
                
                # Check for reasonable ranges
                var_lower = var_name.lower()
                if 'swh' in var_lower or 'significant' in var_lower:
                    print(f"         🌊 SWH Assessment:")
                    reasonable_swh = (finite_vals >= 0) & (finite_vals <= 20)
                    print(f"            Values 0-20m: {reasonable_swh.sum():,} ({reasonable_swh.mean()*100:.1f}%)")
                    if np.any(finite_vals < 0):
                        print(f"            ❌ Negative values: {(finite_vals < 0).sum():,}")
                    if np.any(finite_vals > 20):
                        print(f"            ⚠️  Extreme values >20m: {(finite_vals > 20).sum():,}")
                
                elif 'mwd' in var_lower or 'direction' in var_lower:
                    print(f"         🧭 Wave Direction Assessment:")
                    reasonable_mwd = (finite_vals >= 0) & (finite_vals <= 360)
                    print(f"            Values 0-360°: {reasonable_mwd.sum():,} ({reasonable_mwd.mean()*100:.1f}%)")
                    if np.any(finite_vals < 0):
                        print(f"            ❌ Negative values: {(finite_vals < 0).sum():,}")
                    if np.any(finite_vals > 360):
                        print(f"            ❌ Values >360°: {(finite_vals > 360).sum():,}")
                
                elif 'mwp' in var_lower or 'period' in var_lower:
                    print(f"         ⏱️  Wave Period Assessment:")
                    reasonable_mwp = (finite_vals >= 1) & (finite_vals <= 25)
                    print(f"            Values 1-25s: {reasonable_mwp.sum():,} ({reasonable_mwp.mean()*100:.1f}%)")
                    if np.any(finite_vals < 1):
                        print(f"            ❌ Values <1s: {(finite_vals < 1).sum():,}")
                    if np.any(finite_vals > 25):
                        print(f"            ⚠️  Extreme values >25s: {(finite_vals > 25).sum():,}")
            
            # NaN percentage assessment
            print(f"      🎯 NaN ASSESSMENT:")
            if nan_percent <= 35:
                print(f"         ✅ GOOD: {nan_percent:.1f}% NaN is reasonable for ocean data")
            elif nan_percent <= 50:
                print(f"         ⚠️  MODERATE: {nan_percent:.1f}% NaN is higher than ideal")
            elif nan_percent <= 70:
                print(f"         ❌ HIGH: {nan_percent:.1f}% NaN is concerning")
            else:
                print(f"         🚨 EXCESSIVE: {nan_percent:.1f}% NaN indicates serious issues")
            
            # Spatial analysis if possible
            if len(var_data.dims) >= 2 and lat_coord and lon_coord:
                print(f"      🗺️  SPATIAL ANALYSIS:")
                
                # Get spatial slice (first timestep if 3D)
                if len(var_data.dims) == 3:
                    spatial_slice = var_data.isel({var_data.dims[0]: 0}).values
                else:
                    spatial_slice = var_data.values
                
                # Calculate spatial coverage
                spatial_finite = np.isfinite(spatial_slice)
                spatial_coverage = spatial_finite.sum() / spatial_finite.size * 100
                
                print(f"         Spatial coverage: {spatial_coverage:.1f}%")
                
                # Check by latitude bands
                lat_bands = [
                    (60, 90, "Arctic"),
                    (30, 60, "Northern Mid-lat"),
                    (0, 30, "Northern Tropics"),
                    (-30, 0, "Southern Tropics"),
                    (-60, -30, "Southern Mid-lat"),
                    (-90, -60, "Antarctic")
                ]
                
                for lat_min, lat_max, band_name in lat_bands:
                    lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
                    if len(lat_indices) > 0:
                        band_data = spatial_slice[lat_indices, :]
                        band_coverage = np.isfinite(band_data).mean() * 100
                        print(f"         {band_name:>17} ({lat_min:>3}° to {lat_max:>3}°): {band_coverage:>5.1f}%")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        # Check if this looks like proper ocean data
        wave_vars = [var for var in ds.data_vars if any(pattern in var.lower() 
                    for pattern in ['swh', 'wave', 'significant', 'mwd', 'mwp', 'period', 'direction'])]
        
        if wave_vars:
            print(f"   Wave variables found: {len(wave_vars)}")
            avg_nan_percent = np.mean([
                (np.isnan(ds[var].values).sum() / ds[var].values.size) * 100 
                for var in wave_vars
            ])
            print(f"   Average NaN percentage: {avg_nan_percent:.1f}%")
            
            if avg_nan_percent <= 35:
                print(f"   ✅ DATA QUALITY: EXCELLENT - Normal ocean masking")
            elif avg_nan_percent <= 50:
                print(f"   ⚠️  DATA QUALITY: MODERATE - Some issues present")
            elif avg_nan_percent <= 70:
                print(f"   ❌ DATA QUALITY: POOR - Significant data loss")
            else:
                print(f"   🚨 DATA QUALITY: CRITICAL - Excessive data loss")
        
        print(f"\n📝 RECOMMENDATIONS:")
        if avg_nan_percent > 70:
            print(f"   - Investigate data download parameters")
            print(f"   - Check CDS API query configuration")
            print(f"   - Verify variable names and availability")
        elif avg_nan_percent > 50:
            print(f"   - Check for regional data gaps")
            print(f"   - Verify time period availability")
            print(f"   - Consider alternative data sources")
        else:
            print(f"   - Data quality appears normal for ocean variables")
            print(f"   - Ready for model training")
        
        ds.close()
        
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = "data/era5_global/era5_single_level_201908.nc"
    inspect_ocean_data(file_path)