#!/usr/bin/env python3
"""
Process manually downloaded V1 data from CDS
Simplified for wave-native resolution (0.5°) only
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def diagnose_wave_time_duplication(dataset: xr.Dataset, time_coord: str):
    """Diagnose potential time duplication issues in wave data"""
    
    print(f"\n🔍 DIAGNOSING TIME DUPLICATION ISSUES")
    
    # Get time values
    time_values = dataset[time_coord].values
    time_pandas = pd.to_datetime(time_values)
    
    print(f"   Total timesteps: {len(time_values)}")
    print(f"   Unique timestamps: {len(pd.unique(time_pandas))}")
    
    if len(time_values) != len(pd.unique(time_pandas)):
        print(f"   🚨 TIME DUPLICATION DETECTED!")
        
        unique_times = pd.unique(time_pandas)
        duplication_factor = len(time_values) // len(unique_times)
        
        print(f"   Duplication factor: {duplication_factor}x")
        print(f"   Expected unique times: {len(unique_times)}")
        print(f"   Actual timesteps: {len(time_values)}")
        
        # Check wave variable patterns across duplicated times
        wave_vars = [var for var in dataset.data_vars 
                    if any(pattern in var.lower() for pattern in ['swh', 'wave', 'mwd', 'mwp'])]
        
        if wave_vars:
            print(f"\n   📊 Analyzing wave data across time chunks:")
            
            wave_var = wave_vars[0]  # Use first wave variable
            wave_data = dataset[wave_var]
            
            chunk_size = len(unique_times)
            
            for chunk_idx in range(duplication_factor):
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_size
                
                chunk_data = wave_data.isel({time_coord: slice(start_idx, end_idx)})
                finite_percent = (np.isfinite(chunk_data.values).sum() / chunk_data.values.size) * 100
                
                print(f"      Chunk {chunk_idx} (times {start_idx}-{end_idx-1}): {finite_percent:.1f}% valid data")
        
        return True, duplication_factor, unique_times
    else:
        print(f"   ✅ No time duplication detected")
        return False, 1, time_pandas

def deduplicate_wave_data(dataset: xr.Dataset, time_coord: str, duplication_factor: int, unique_times):
    """Remove time duplication from wave data by selecting best chunk"""
    
    print(f"\n🔧 DEDUPLICATING WAVE DATA")
    
    # Find the chunk with the most valid wave data
    wave_vars = [var for var in dataset.data_vars 
                if any(pattern in var.lower() for pattern in ['swh', 'wave', 'mwd', 'mwp'])]
    
    if not wave_vars:
        print(f"   ❌ No wave variables found for deduplication")
        return dataset
    
    chunk_size = len(unique_times)
    best_chunk_idx = 0
    best_chunk_score = 0
    
    print(f"   🎯 Finding best time chunk...")
    
    for chunk_idx in range(duplication_factor):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        
        chunk_score = 0
        chunk_valid_vars = 0
        
        for wave_var in wave_vars:
            chunk_data = dataset[wave_var].isel({time_coord: slice(start_idx, end_idx)})
            finite_percent = (np.isfinite(chunk_data.values).sum() / chunk_data.values.size) * 100
            
            if finite_percent > 10:  # At least 10% valid data
                chunk_score += finite_percent
                chunk_valid_vars += 1
        
        avg_score = chunk_score / len(wave_vars) if wave_vars else 0
        
        print(f"      Chunk {chunk_idx}: {avg_score:.1f}% avg valid, {chunk_valid_vars}/{len(wave_vars)} vars with data")
        
        if avg_score > best_chunk_score:
            best_chunk_score = avg_score
            best_chunk_idx = chunk_idx
    
    print(f"   ✅ Selected chunk {best_chunk_idx} with {best_chunk_score:.1f}% average valid data")
    
    # Extract the best chunk
    best_start_idx = best_chunk_idx * chunk_size
    best_end_idx = best_start_idx + chunk_size
    
    print(f"   📤 Extracting timesteps {best_start_idx} to {best_end_idx-1}")
    
    # Create deduplicated dataset
    deduplicated_vars = {}
    
    for var_name, var_data in dataset.data_vars.items():
        if time_coord in var_data.dims:
            # Extract best time chunk
            deduplicated_var = var_data.isel({time_coord: slice(best_start_idx, best_end_idx)})
            deduplicated_vars[var_name] = deduplicated_var
        else:
            # Keep variables without time dimension as-is
            deduplicated_vars[var_name] = var_data
    
    # Create new time coordinate
    new_time_values = dataset[time_coord].values[best_start_idx:best_end_idx]
    
    # Create new coordinates
    new_coords = dict(dataset.coords)
    new_coords[time_coord] = (time_coord, new_time_values)
    
    # Create deduplicated dataset
    deduplicated_ds = xr.Dataset(deduplicated_vars, coords=new_coords)
    
    # Copy attributes
    deduplicated_ds.attrs.update(dataset.attrs)
    deduplicated_ds.attrs.update({
        'time_deduplication_applied': 'true',  # Convert boolean to string
        'original_timesteps': str(len(dataset[time_coord])),  # Convert to string
        'deduplicated_timesteps': str(len(new_time_values)),  # Convert to string
        'selected_chunk': str(best_chunk_idx),                # Convert to string
        'duplication_factor_removed': str(duplication_factor) # Convert to string
    })
    
    print(f"   ✅ Deduplication complete:")
    print(f"      Original timesteps: {len(dataset[time_coord])}")
    print(f"      Deduplicated timesteps: {len(new_time_values)}")
    
    return deduplicated_ds

def validate_v1_data_quality(dataset: xr.Dataset):
    """Validate V1 data quality"""
    
    print(f"   🔍 Data quality validation...")
    
    # Get time and spatial dimensions
    time_coord = 'time' if 'time' in dataset.dims else 'valid_time'
    lat_coord = 'latitude' if 'latitude' in dataset.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in dataset.coords else 'lon'
    
    if lat_coord in dataset.coords and lon_coord in dataset.coords:
        lats = dataset[lat_coord].values
        lons = dataset[lon_coord].values
        total_spatial_points = len(lats) * len(lons)
        
        print(f"      Grid: {len(lats)} × {len(lons)} = {total_spatial_points:,} points")
        
        # Expected ocean coverage (~71% of Earth)
        expected_ocean_percent = 71
        expected_land_percent = 29
        
        print(f"      Expected for ocean variables: ~{expected_land_percent}% NaN (land)")
    
    # Analyze key variable groups
    variable_groups = {
        'wave_vars': ['swh', 'mwd', 'mwp', 'shww', 'significant_height_of_combined_wind_waves_and_swell', 
                      'mean_wave_direction', 'mean_wave_period', 'significant_height_of_wind_waves'],
        'surface_vars': ['u10', 'v10', 'msl', 'sst', 'tp', '10m_u_component_of_wind', 
                        '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_surface_temperature'],
        'atmo_vars': [var for var in dataset.data_vars if '_850' in var or '_500' in var]
    }
    
    for group_name, var_patterns in variable_groups.items():
        print(f"\n      📊 {group_name.upper()}:")
        
        if group_name == 'atmo_vars':
            group_vars = var_patterns  # Already filtered
        else:
            group_vars = []
            for var_name in dataset.data_vars:
                if any(pattern in var_name for pattern in var_patterns):
                    group_vars.append(var_name)
        
        if not group_vars:
            print(f"         ❌ No variables found for {group_name}")
            continue
        
        print(f"         Variables: {len(group_vars)}")
        
        for var_name in group_vars[:5]:  # Show first 5
            values = dataset[var_name].values
            total_points = values.size
            nan_count = np.isnan(values).sum()
            nan_percent = (nan_count / total_points) * 100
            
            finite_vals = values[np.isfinite(values)]
            if len(finite_vals) > 0:
                val_range = f"[{np.min(finite_vals):.3f}, {np.max(finite_vals):.3f}]"
            else:
                val_range = "All NaN"
            
            # Assess quality based on variable type
            if group_name == 'wave_vars':
                quality = "✅ Good" if nan_percent <= 50 else "⚠️ High" if nan_percent <= 70 else "❌ Poor"
            else:
                quality = "✅ Good" if nan_percent <= 10 else "⚠️ High" if nan_percent <= 30 else "❌ Poor"
            
            print(f"         {var_name:>25}: {nan_percent:>5.1f}% NaN, {val_range} {quality}")
        
        if len(group_vars) > 5:
            print(f"         ... and {len(group_vars) - 5} more variables")
    
    # Latitude band analysis for wave variables
    if lat_coord in dataset.coords:
        print(f"\n      🌍 Latitude band analysis (wave variables):")
        
        wave_var = None
        for var_name in dataset.data_vars:
            if any(pattern in var_name for pattern in ['swh', 'significant_height']):
                wave_var = var_name
                break
        
        if wave_var and len(dataset[wave_var].dims) >= 2:
            wave_data = dataset[wave_var]
            
            # Get spatial slice (first time if 3D)
            if len(wave_data.dims) == 3:
                spatial_data = wave_data.isel({time_coord: 0}).values
            else:
                spatial_data = wave_data.values
            
            # Define latitude bands
            lat_bands = [
                (60, 90, "Arctic"),
                (30, 60, "Northern Mid-lat"),
                (0, 30, "Northern Tropics"), 
                (-30, 0, "Southern Tropics"),
                (-60, -30, "Southern Mid-lat"),
                (-90, -60, "Antarctic")
            ]
            
            for lat_min, lat_max, band_name in lat_bands:
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                if np.any(lat_mask):
                    band_data = spatial_data[lat_mask, :]
                    coverage = np.isfinite(band_data).mean() * 100
                    
                    status = "✅" if coverage >= 50 else "⚠️" if coverage >= 30 else "❌"
                    print(f"         {band_name:>17} ({lat_min:>3}° to {lat_max:>3}°): {coverage:>5.1f}% {status}")
    
    print(f"      ✅ Quality validation complete")

def process_v1_manual_data(year_month: str, data_root: str = "data/v1_global"):
    """
    Process manually downloaded V1 data - WAVE NATIVE RESOLUTION ONLY
    Downscales atmospheric data to 0.5° wave resolution
    """
    
    print(f"🔧 PROCESSING V1 MANUAL DATA: {year_month}")
    print(f"   Target: Wave native resolution (0.5°)")
    print("=" * 60)
    
    data_root = Path(data_root)
    era5_root = data_root / "era5"
    processed_root = data_root / "processed"
    processed_root.mkdir(exist_ok=True)
    
    # Input file paths
    atmo_file = era5_root / f"era5_pressure_{year_month}.nc"
    single_level_dir = era5_root / f"{year_month}_single_level"
    
    # Expected single-level files from CDS split
    single_level_files = {
        'accum': single_level_dir / "data_stream-oper_stepType-accum.nc",
        'instant': single_level_dir / "data_stream-oper_stepType-instant.nc", 
        'wave': single_level_dir / "data_stream-wave_stepType-instant.nc"
    }
    
    # Output file
    output_file = processed_root / f"v1_era5_{year_month}.nc"
    
    if output_file.exists():
        print(f"✅ Output already exists: {output_file.name}")
        return output_file
    
    # Validate input files
    print(f"📋 Validating input files...")
    
    if not atmo_file.exists():
        print(f"❌ Missing atmospheric file: {atmo_file}")
        return None
    
    missing_files = []
    for stream_name, file_path in single_level_files.items():
        if not file_path.exists():
            missing_files.append(f"{stream_name}: {file_path}")
        else:
            print(f"   ✅ {stream_name}: {file_path.name}")
    
    if missing_files:
        print(f"❌ Missing single-level files:")
        for missing in missing_files:
            print(f"   {missing}")
        return None
    
    print(f"   ✅ Atmospheric: {atmo_file.name}")
    
    try:
        # Step 1: Load single-level data (wave resolution)
        print(f"\n📥 STEP 1: Loading single-level data")
        
        single_datasets = {}
        for stream_name, file_path in single_level_files.items():
            print(f"   📖 Loading {stream_name} data...")
            ds = xr.open_dataset(file_path)
            single_datasets[stream_name] = ds
            print(f"      Dimensions: {dict(ds.dims)}")
            print(f"      Variables: {list(ds.data_vars.keys())}")
        
        # Step 2: Concatenate single-level data streams
        print(f"\n🔗 STEP 2: Concatenating single-level data streams")
        
        # Find common time coordinate
        time_coord = None
        for coord in ['time', 'valid_time']:
            if coord in single_datasets['wave'].coords:
                time_coord = coord
                break
        
        if time_coord is None:
            print(f"❌ No time coordinate found in wave data")
            return None
        
        print(f"   Time coordinate: {time_coord}")
        
        # Get reference time from wave data (most reliable)
        reference_times = single_datasets['wave'][time_coord].values
        wave_ds = single_datasets['wave']
        
        # Get wave resolution info
        lat_coord = 'latitude' if 'latitude' in wave_ds.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in wave_ds.coords else 'lon'
        
        wave_lats = wave_ds[lat_coord].values
        wave_lons = wave_ds[lon_coord].values
        
        print(f"   Wave resolution: {len(wave_lats)} × {len(wave_lons)} (0.5°)")
        print(f"   Reference time steps: {len(reference_times)}")
        
        # Align all single-level datasets to reference time
        aligned_datasets = {}
        
        for stream_name, ds in single_datasets.items():
            print(f"   🔧 Aligning {stream_name} data...")
            
            # Check if times match
            ds_times = ds[time_coord].values
            
            if np.array_equal(ds_times, reference_times):
                print(f"      ✅ Times already aligned")
                aligned_datasets[stream_name] = ds
            else:
                print(f"      🔄 Reindexing to reference times")
                # Reindex to reference times
                aligned_ds = ds.reindex({time_coord: reference_times}, method='nearest', tolerance='1H')
                aligned_datasets[stream_name] = aligned_ds
        
        # Combine all single-level variables
        print(f"   🎯 Combining single-level variables...")
        
        combined_vars = {}
        
        for stream_name, ds in aligned_datasets.items():
            for var_name in ds.data_vars:
                if var_name in combined_vars:
                    print(f"      ⚠️  Duplicate variable {var_name} in {stream_name}")
                else:
                    combined_vars[var_name] = ds[var_name]
                    print(f"      ✅ {var_name} from {stream_name}")
        
        print(f"   📊 Total single-level variables: {len(combined_vars)}")
        
        # Create wave-resolution coordinates
        wave_coords = {
            lat_coord: wave_ds[lat_coord],
            lon_coord: wave_ds[lon_coord],
            time_coord: wave_ds[time_coord]
        }
        
        # Create combined single-level dataset at wave resolution
        single_combined = xr.Dataset(combined_vars, coords=wave_coords)
        
        # Step 3: Load and process atmospheric data
        print(f"\n🏔️  STEP 3: Processing atmospheric data")
        
        # Load atmospheric data
        print(f"   📖 Loading atmospheric data...")
        atmo_ds = xr.open_dataset(atmo_file)
        print(f"      Dimensions: {dict(atmo_ds.dims)}")
        
        # Align atmospheric time to reference
        atmo_time_coord = None
        for coord in ['time', 'valid_time']:
            if coord in atmo_ds.coords:
                atmo_time_coord = coord
                break
        
        atmo_times = atmo_ds[atmo_time_coord].values
        if not np.array_equal(atmo_times, reference_times):
            print(f"   🔄 Aligning atmospheric times...")
            atmo_ds = atmo_ds.reindex({atmo_time_coord: reference_times}, method='nearest', tolerance='1H')
        
        # Find pressure coordinate and extract levels
        pressure_coord = None
        for coord in ['level', 'pressure_level', 'plev']:
            if coord in atmo_ds.coords:
                pressure_coord = coord
                break
        
        pressure_levels = atmo_ds[pressure_coord].values
        target_levels = [850, 500]
        
        print(f"   🎯 Extracting and regridding atmospheric variables...")
        print(f"      Available levels: {pressure_levels}")
        print(f"      Target levels: {target_levels}")
        
        # Extract and regrid atmospheric variables
        atmo_vars = {}
        
        for var_name in atmo_ds.data_vars:
            var_data = atmo_ds[var_name]
            
            if pressure_coord in var_data.dims:
                print(f"   🔧 Processing {var_name}...")
                
                for level in target_levels:
                    if level in pressure_levels:
                        try:
                            # Extract level data
                            level_data = var_data.sel({pressure_coord: level}).drop_vars(pressure_coord)
                            
                            # Regrid to wave resolution
                            print(f"      🔄 Regridding {var_name}_{level} to wave resolution...")
                            
                            # Find atmospheric coordinates
                            atmo_lat_coord = 'latitude' if 'latitude' in level_data.coords else 'lat'
                            atmo_lon_coord = 'longitude' if 'longitude' in level_data.coords else 'lon'
                            
                            # Interpolate to wave resolution
                            regridded = level_data.interp({
                                atmo_lat_coord: wave_lats,
                                atmo_lon_coord: wave_lons
                            }, method='linear')
                            
                            # Rename coordinates to match wave data
                            if atmo_lat_coord != lat_coord or atmo_lon_coord != lon_coord:
                                regridded = regridded.rename({
                                    atmo_lat_coord: lat_coord,
                                    atmo_lon_coord: lon_coord
                                })
                            
                            atmo_var_name = f"{var_name}_{level}"
                            atmo_vars[atmo_var_name] = regridded
                            print(f"      ✅ {atmo_var_name} regridded to wave resolution")
                            
                        except Exception as e:
                            print(f"      ❌ Failed to process {var_name}@{level}: {e}")
                    else:
                        print(f"      ⚠️  Level {level} hPa not available for {var_name}")
        
        print(f"   📊 Processed atmospheric variables: {len(atmo_vars)}")
        
        # Step 4: Combine everything at wave resolution
        print(f"\n🎯 STEP 4: Creating final V1 dataset at wave resolution")
        
        # Combine all variables
        final_vars = {}
        final_vars.update(single_combined.data_vars)  # Wave + surface at 0.5°
        final_vars.update(atmo_vars)                  # Atmospheric regridded to 0.5°
        
        # Create final dataset with wave coordinates
        v1_dataset = xr.Dataset(final_vars, coords=wave_coords)
        
        # Add metadata
        v1_dataset.attrs.update({
            'title': f'V1 Global Training Data {year_month}',
            'source': 'ERA5 reanalysis - wave native resolution',
            'processing_date': pd.Timestamp.now().isoformat(),
            'model_version': 'V1',
            'target_resolution': 'wave_native_0.5_degrees',
            'grid_size': f"{len(wave_lats)}x{len(wave_lons)}",
            'single_level_variables': str(len(single_combined.data_vars)),
            'atmospheric_variables': str(len(atmo_vars)),
            'total_variables': str(len(final_vars)),
            'pressure_levels_used': str(target_levels)
        })
        
        print(f"   📊 Final V1 dataset:")
        print(f"      Variables: {len(final_vars)}")
        print(f"      Dimensions: {dict(v1_dataset.dims)}")
        print(f"      Grid: {len(wave_lats)} × {len(wave_lons)} (wave resolution)")
        
        # Step 5: Check for and fix time duplication
        print(f"\n🕐 STEP 5: Check for time duplication issues")
        
        has_duplication, duplication_factor, unique_times = diagnose_wave_time_duplication(v1_dataset, time_coord)
        
        if has_duplication:
            print(f"   🔧 Applying deduplication...")
            v1_dataset_clean = deduplicate_wave_data(v1_dataset, time_coord, duplication_factor, unique_times)
            v1_dataset.close()
            v1_dataset = v1_dataset_clean
        else:
            print(f"   ✅ No deduplication needed")
        
        # Step 6: Data quality validation
        print(f"\n🔍 STEP 6: Data quality validation")
        validate_v1_data_quality(v1_dataset)
        
        # Step 7: Save processed data
        print(f"\n💾 STEP 7: Saving V1 dataset")
        
        # Use compression
        encoding = {var: {'zlib': True, 'complevel': 1} for var in final_vars}
        v1_dataset.to_netcdf(output_file, encoding=encoding)
        
        print(f"   ✅ V1 dataset saved: {output_file.name}")
        print(f"   📏 File size: {output_file.stat().st_size / (1024**2):.1f} MB")
        
        # Final verification
        with xr.open_dataset(output_file) as verify_ds:
            print(f"   ✅ Verification: {len(verify_ds.data_vars)} variables, {len(verify_ds[time_coord])} time steps")
            print(f"   📊 Final grid: {dict(verify_ds.dims)}")
        
        # Cleanup
        atmo_ds.close()
        for ds in single_datasets.values():
            ds.close()
        for ds in aligned_datasets.values():
            if ds not in single_datasets.values():
                ds.close()
        v1_dataset.close()
        
        return output_file
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main processing function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process manually downloaded V1 data - Wave native only')
    parser.add_argument('--year_month', type=str, help='Process specific year_month (e.g., 202101)')
    parser.add_argument('--data_root', default='data/v1_global', help='Data root directory')
    parser.add_argument('--validate', type=str, help='Validate specific processed file')
    
    args = parser.parse_args()
    
    if args.validate:
        print(f"🔍 VALIDATING: {args.validate}")
        try:
            with xr.open_dataset(args.validate) as ds:
                validate_v1_data_quality(ds)
        except Exception as e:
            print(f"❌ Validation failed: {e}")
        return
    
    if args.year_month:
        result = process_v1_manual_data(args.year_month, args.data_root)
        if result:
            print(f"\n🎉 SUCCESS: {result.name}")
            print(f"   Resolution: Wave native (0.5°)")
            print(f"   Ready for V1 model training!")
        else:
            print(f"\n❌ FAILED to process {args.year_month}")
    else:
        print("❌ Specify --year_month")
        print("Example: python process_v1_manual_data.py --year_month 202101")

if __name__ == "__main__":
    main()