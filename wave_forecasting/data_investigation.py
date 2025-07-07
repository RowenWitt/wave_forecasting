#!/usr/bin/env python3
"""
Investigate and fix the data loading issues
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def investigate_wave_data_issue():
    """Investigate why wave data is all NaN"""
    
    print("🔍 INVESTIGATING WAVE DATA ISSUE")
    print("=" * 40)
    
    # Setup data loading
    from data.loaders import ERA5DataManager, GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    era5_manager = ERA5DataManager(data_config)
    gebco_manager = GEBCODataManager(data_config)
    
    print(f"📁 Available ERA5 files:")
    print(f"   Atmospheric: {len(era5_manager.available_files['atmospheric'])}")
    print(f"   Wave: {len(era5_manager.available_files['wave'])}")
    
    # Load raw data to check
    first_file = era5_manager.available_files['atmospheric'][0]
    filename = Path(first_file).stem
    year_month = filename.split('_')[-1]
    year, month = int(year_month[:4]), int(year_month[4:6])
    
    print(f"   Using: {year}-{month:02d}")
    
    # Check raw ERA5 data
    print(f"\n📊 RAW ERA5 DATA CHECK:")
    era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
    
    print(f"   Atmospheric data: {era5_atmo}")
    print(f"   Wave data: {era5_waves}")
    
    # Check what variables exist in wave data
    print(f"\n🌊 WAVE DATA VARIABLES:")
    for var in era5_waves.data_vars:
        data = era5_waves[var]
        print(f"   {var}: {data.shape}")
        
        # Check for NaN values in the raw data
        if hasattr(data, 'values'):
            values = data.values
            nan_count = np.isnan(values).sum()
            total_count = values.size
            print(f"     NaN values: {nan_count}/{total_count} ({100*nan_count/total_count:.1f}%)")
            
            if nan_count < total_count:  # Some valid data exists
                valid_values = values[~np.isnan(values)]
                print(f"     Valid range: {valid_values.min():.3f} to {valid_values.max():.3f}")
    
    # Check specific time indices
    print(f"\n⏰ TIME INDEX CHECK:")
    total_times = len(era5_waves.valid_time)
    print(f"   Total time steps: {total_times}")
    
    # Check time indices around our test range (20-25)
    test_indices = [15, 20, 21, 25, 30] if total_times > 30 else list(range(min(5, total_times)))
    
    for t_idx in test_indices:
        if t_idx < total_times:
            print(f"   Time {t_idx}: {era5_waves.valid_time[t_idx].values}")
            
            # Check wave data at this time
            for var in ['swh', 'mwd', 'mwp']:
                if var in era5_waves:
                    data_slice = era5_waves[var].isel(valid_time=t_idx)
                    values = data_slice.values
                    nan_count = np.isnan(values).sum()
                    total_count = values.size
                    
                    if nan_count < total_count:
                        valid_values = values[~np.isnan(values)]
                        print(f"     {var}: {nan_count}/{total_count} NaN, range {valid_values.min():.3f}-{valid_values.max():.3f}")
                    else:
                        print(f"     {var}: ALL NaN")
    
    return era5_atmo, era5_waves, year, month

def test_interpolation_process(era5_atmo, era5_waves, year, month):
    """Test the interpolation process step by step"""
    
    print(f"\n🔧 TESTING INTERPOLATION PROCESS")
    print("=" * 35)
    
    from data.loaders import GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    # Load GEBCO
    gebco_manager = GEBCODataManager(data_config)
    gebco_data = gebco_manager.load_bathymetry()
    
    print(f"✅ GEBCO data loaded")
    
    # Create mesh
    mesh = IcosahedralMesh(mesh_config)
    print(f"✅ Mesh created: {mesh.vertices.shape[0]} vertices")
    
    # Create interpolator
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
    print(f"✅ Interpolator created")
    
    # Test interpolation at a specific time
    test_time = 0  # Start with the first time index
    
    print(f"\n🎯 TESTING TIME INDEX {test_time}:")
    
    # Get mesh coordinates
    region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
    regional_coords = mesh.vertices[region_indices]
    lats = regional_coords[:, 0]
    lons = regional_coords[:, 1]
    
    print(f"   Regional nodes: {len(region_indices)}")
    print(f"   Lat range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"   Lon range: {lons.min():.2f} to {lons.max():.2f}")
    
    # Test interpolation
    try:
        interpolated_data = interpolator.interpolate_to_points(lats, lons, test_time)
        
        print(f"   ✅ Interpolation successful")
        print(f"   Variables: {list(interpolated_data.keys())}")
        
        # Check wave variables specifically
        for var in ['swh', 'mwd', 'mwp']:
            if var in interpolated_data:
                values = interpolated_data[var]
                nan_count = np.isnan(values).sum()
                total_count = values.size
                
                print(f"   {var}: {nan_count}/{total_count} NaN", end="")
                if nan_count < total_count:
                    valid_values = values[~np.isnan(values)]
                    print(f", range {valid_values.min():.3f}-{valid_values.max():.3f}")
                else:
                    print(" (ALL NaN)")
        
        return interpolated_data, region_indices
        
    except Exception as e:
        print(f"   ❌ Interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_mesh_data_loader(era5_atmo, era5_waves, interpolated_data, region_indices):
    """Test the MeshDataLoader process"""
    
    if interpolated_data is None:
        print("⚠️  Skipping MeshDataLoader test - interpolation failed")
        return
    
    print(f"\n📊 TESTING MESH DATA LOADER")
    print("=" * 30)
    
    from data.loaders import GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    gebco_manager = GEBCODataManager(data_config)
    gebco_data = gebco_manager.load_bathymetry()
    
    mesh = IcosahedralMesh(mesh_config)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
    mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
    
    # Test loading features at different time indices
    test_times = [0, 1, 2, 5, 10, 15, 20] if len(era5_waves.valid_time) > 20 else [0, 1, 2]
    
    for t_idx in test_times:
        if t_idx < len(era5_waves.valid_time):
            try:
                features_data = mesh_loader.load_features(time_idx=t_idx)
                features = features_data['features']
                feature_names = features_data['feature_names']
                
                print(f"   Time {t_idx}: Shape {features.shape}")
                
                # Check wave variables
                wave_indices = [3, 4, 5]  # swh, mwd, mwp
                for i, wave_idx in enumerate(wave_indices):
                    if wave_idx < features.shape[1]:
                        wave_values = features[:, wave_idx]
                        nan_count = np.isnan(wave_values).sum()
                        total_count = wave_values.size
                        
                        var_name = feature_names[wave_idx] if wave_idx < len(feature_names) else f"var_{wave_idx}"
                        
                        print(f"     {var_name}: {nan_count}/{total_count} NaN", end="")
                        if nan_count < total_count:
                            valid_values = wave_values[~np.isnan(wave_values)]
                            print(f", range {valid_values.min():.3f}-{valid_values.max():.3f}")
                        else:
                            print(" (ALL NaN)")
                
            except Exception as e:
                print(f"   Time {t_idx}: ❌ Failed - {e}")

def find_valid_time_indices():
    """Find time indices that have valid wave data"""
    
    print(f"\n🔍 FINDING VALID TIME INDICES")
    print("=" * 35)
    
    era5_atmo, era5_waves, year, month = investigate_wave_data_issue()
    
    valid_times = []
    total_times = len(era5_waves.valid_time)
    
    # Check each time index
    for t_idx in range(min(total_times, 50)):  # Check first 50 time steps
        has_valid_data = False
        
        for var in ['swh', 'mwd', 'mwp']:
            if var in era5_waves:
                data_slice = era5_waves[var].isel(valid_time=t_idx)
                values = data_slice.values
                
                if not np.all(np.isnan(values)):
                    has_valid_data = True
                    break
        
        if has_valid_data:
            valid_times.append(t_idx)
            
        if t_idx % 10 == 0:
            print(f"   Checked time {t_idx}: {'✅' if has_valid_data else '❌'}")
    
    print(f"\n📊 VALID TIME SUMMARY:")
    print(f"   Valid time indices: {valid_times[:10]}{'...' if len(valid_times) > 10 else ''}")
    print(f"   Total valid times: {len(valid_times)}/{total_times}")
    
    if valid_times:
        print(f"   Recommendation: Use time indices {valid_times[5:15]} for evaluation")
        return valid_times
    else:
        print(f"   ❌ No valid wave data found!")
        return []

def main():
    """Main data investigation"""
    
    print("🔍 WAVE DATA INVESTIGATION AND FIX")
    print("=" * 50)
    
    try:
        # Step 1: Check raw data
        era5_atmo, era5_waves, year, month = investigate_wave_data_issue()
        
        # Step 2: Test interpolation 
        interpolated_data, region_indices = test_interpolation_process(era5_atmo, era5_waves, year, month)
        
        # Step 3: Test mesh data loader
        test_mesh_data_loader(era5_atmo, era5_waves, interpolated_data, region_indices)
        
        # Step 4: Find valid time indices
        valid_times = find_valid_time_indices()
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if valid_times:
            print(f"   ✅ Use time indices: {valid_times[5:10]} instead of [20, 21]")
            print(f"   ✅ Update ForecastConfig.initial_time_idx to {valid_times[5]}")
            print(f"   ✅ Re-run evaluation with valid data")
        else:
            print(f"   ❌ No valid wave data found - check ERA5 download")
            print(f"   ❌ May need to re-download wave data for {year}-{month:02d}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Use valid time indices for evaluation")
        print(f"   2. Check model performance with valid ground truth")
        print(f"   3. If still high RMSE, investigate model training data format")
        
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()