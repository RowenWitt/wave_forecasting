"""
Clean Data Integration Script - Real Data Only
Process ERA5 data to include climate anomalies and bathymetry for V1 training
FAILS if real climatology or bathymetry data is missing
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def compute_climate_anomalies(era5_data: xr.Dataset, month: int, 
                             climatology: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
    """Compute climate anomalies from ERA5 data"""
    
    print(f"   📊 Computing climate anomalies for month {month}...")
    
    anomalies = {}
    
    # Variable mapping
    var_mapping = {
        'sst': 'sea_surface_temperature',
        'slp': 'mean_sea_level_pressure', 
        'u850': 'u_component_of_wind_850',
        'v850': 'v_component_of_wind_850',
        'precip': 'total_precipitation',
        'z500': 'geopotential_500'
    }
    
    for clim_var, era5_var in var_mapping.items():
        try:
            if era5_var not in era5_data.data_vars:
                print(f"      ⚠️  ERA5 variable {era5_var} not found, skipping {clim_var} anomaly")
                continue
                
            if clim_var not in climatology:
                print(f"      ⚠️  No climatology for {clim_var}, skipping anomaly")
                continue
            
            clim_ds = climatology[clim_var]
            if clim_ds is None:
                print(f"      ⚠️  Climatology for {clim_var} is None, skipping")
                continue
            
            # Get ERA5 data
            era5_field = era5_data[era5_var]
            
            # Find the climatology variable in the dataset
            clim_var_name = None
            possible_names = [clim_var, era5_var, f'{clim_var}_climatology', f'{clim_var}_clim']
            
            for name in possible_names:
                if name in clim_ds.data_vars:
                    clim_var_name = name
                    break
            
            if clim_var_name is None:
                print(f"      ❌ Cannot find {clim_var} variable in climatology dataset")
                print(f"         Available variables: {list(clim_ds.data_vars.keys())}")
                continue
            
            # Get climatology for this month
            clim_field = clim_ds[clim_var_name]
            
            if 'month' in clim_ds.coords:
                if month in clim_ds.month.values:
                    clim_field = clim_field.sel(month=month)
                else:
                    print(f"      ❌ Month {month} not found in climatology, available: {clim_ds.month.values}")
                    continue
            else:
                print(f"      ⚠️  No monthly dimension in {clim_var} climatology, using annual mean")
            
            # Standardize coordinate names
            if 'latitude' in clim_field.coords and 'longitude' in clim_field.coords:
                clim_field = clim_field.rename({'latitude': 'lat', 'longitude': 'lon'})
            
            # Check if grids match
            era5_lats = era5_field.lat.values
            era5_lons = era5_field.lon.values
            clim_lats = clim_field.lat.values  
            clim_lons = clim_field.lon.values
            
            grids_match = (np.array_equal(era5_lats, clim_lats) and 
                          np.array_equal(era5_lons, clim_lons))
            
            if not grids_match:
                print(f"         Interpolating {clim_var} climatology to ERA5 grid...")
                print(f"         ERA5 grid: {len(era5_lats)} x {len(era5_lons)}")
                print(f"         Climatology grid: {len(clim_lats)} x {len(clim_lons)}")
                
                clim_field = clim_field.interp(
                    lat=era5_lats,
                    lon=era5_lons,
                    method='linear'
                )
            
            # Compute anomaly
            anomaly = era5_field - clim_field
            anomalies[f'{clim_var}_anomaly'] = anomaly
            
            # Statistics
            anom_min = float(anomaly.min().values)
            anom_max = float(anomaly.max().values)
            anom_std = float(anomaly.std().values)
            
            print(f"      ✅ {clim_var}_anomaly: range=[{anom_min:.3f}, {anom_max:.3f}], std={anom_std:.3f}")
            
        except Exception as e:
            print(f"      ❌ Error computing {clim_var} anomaly: {e}")
            continue
    
    print(f"   ✅ Successfully computed {len(anomalies)} climate anomalies")
    return anomalies

def process_gebco_bathymetry(gebco_path: str, target_resolution: float = 0.5) -> xr.DataArray:
    """Process GEBCO bathymetry to target resolution - with caching"""
    
    print(f"🌊 Processing GEBCO 2023 bathymetry to {target_resolution}° resolution...")
    
    # Check for cached processed file first
    cache_file = Path(gebco_path) / f"gebco_2023_{target_resolution}deg_processed.nc"
    
    if cache_file.exists():
        print(f"   📂 Loading cached processed bathymetry: {cache_file.name}")
        try:
            ocean_depth = xr.open_dataarray(cache_file)
            print(f"   ✅ Cached bathymetry loaded:")
            print(f"      Depth range: {ocean_depth.min().values:.1f} to {ocean_depth.max().values:.1f}m")
            print(f"      Grid: {ocean_depth.shape}")
            return ocean_depth
        except Exception as e:
            print(f"   ⚠️  Error loading cached file: {e}")
            print(f"   🔄 Will reprocess...")
    
    # Your actual GEBCO file
    gebco_file = Path(gebco_path) / "GEBCO_2023.nc"
    
    if not gebco_file.exists():
        raise FileNotFoundError(f"GEBCO file not found: {gebco_file}. Cannot proceed without real bathymetry data.")
    
    try:
        print(f"   📂 Loading GEBCO 2023 from {gebco_file.name}")
        file_size_gb = gebco_file.stat().st_size / 1e9
        print(f"      File size: {file_size_gb:.1f} GB")
        
        # Open GEBCO dataset
        print(f"      Opening GEBCO dataset...")
        gebco_ds = xr.open_dataset(gebco_file)
        
        # GEBCO 2023 standard variable names
        lat_coord = 'lat' if 'lat' in gebco_ds.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in gebco_ds.coords else 'longitude' 
        depth_var = 'elevation' if 'elevation' in gebco_ds.data_vars else 'z'
        
        if depth_var not in gebco_ds.data_vars:
            available_vars = list(gebco_ds.data_vars.keys())
            raise ValueError(f"Cannot find bathymetry variable in GEBCO. Available: {available_vars}")
        
        print(f"      GEBCO info:")
        print(f"         Variables: {list(gebco_ds.data_vars.keys())}")
        print(f"         Coordinates: {list(gebco_ds.coords.keys())}")
        print(f"         Shape: {dict(gebco_ds.sizes)}")
        
        # Extract elevation/bathymetry
        elevation = gebco_ds[depth_var]
        
        # Rename coordinates for consistency
        elevation = elevation.rename({lat_coord: 'lat', lon_coord: 'lon'})
        
        # Ensure longitude is 0-360 (if it's -180 to 180)
        if elevation.lon.min() < 0:
            print(f"      Converting longitude from [-180,180] to [0,360]...")
            elevation = elevation.assign_coords(lon=(elevation.lon % 360))
            elevation = elevation.sortby('lon')
        
        # Create target grid at specified resolution
        target_lats = np.arange(-90, 90 + target_resolution, target_resolution)
        target_lons = np.arange(0, 360, target_resolution)
        
        print(f"   🔄 Processing to {target_resolution}° resolution...")
        print(f"      Original grid: {elevation.sizes['lat']} x {elevation.sizes['lon']} = {elevation.size:,} points")
        print(f"      Target grid: {len(target_lats)} x {len(target_lons)} = {len(target_lats) * len(target_lons):,} points")
        reduction_factor = elevation.size / (len(target_lats) * len(target_lons))
        print(f"      Memory reduction: {reduction_factor:.0f}x smaller")
        
        # If GEBCO is very high resolution, coarsen first for efficiency
        if elevation.sizes['lat'] > 5000:
            coarsen_factor = max(1, elevation.sizes['lat'] // 2000)
            print(f"      Pre-coarsening by factor {coarsen_factor} (this takes time but saves for future)...")
            elevation = elevation.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            print(f"      After coarsening: {elevation.sizes['lat']} x {elevation.sizes['lon']}")
        
        # Interpolate to target resolution
        print(f"      Interpolating to final {target_resolution}° grid...")
        elevation_interp = elevation.interp(
            lat=target_lats,
            lon=target_lons,
            method='linear'
        )
        
        # Convert to ocean depth (negative elevation = positive depth)
        ocean_depth = -elevation_interp.where(elevation_interp < 0, 0)
        ocean_depth = ocean_depth.rename('ocean_depth')
        
        # Add metadata
        ocean_depth.attrs.update({
            'long_name': 'Ocean depth from GEBCO 2023',
            'units': 'meters',
            'source': 'GEBCO_2023.nc',
            'processing': f'Interpolated to {target_resolution}° resolution',
            'positive': 'down',
            'creation_date': datetime.now().isoformat(),
            'target_resolution': target_resolution
        })
        
        # Save processed bathymetry for future use
        print(f"   💾 Saving processed bathymetry to cache...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ocean_depth.to_netcdf(cache_file)
        
        print(f"   ✅ GEBCO processed and cached successfully:")
        print(f"      Depth range: {ocean_depth.min().values:.1f} to {ocean_depth.max().values:.1f}m")
        print(f"      Ocean points: {(ocean_depth > 0).sum().values:,}")
        print(f"      Land points: {(ocean_depth == 0).sum().values:,}")
        print(f"      Final grid: {ocean_depth.shape}")
        print(f"      Cache saved: {cache_file.name}")
        
        gebco_ds.close()
        return ocean_depth
        
    except Exception as e:
        raise RuntimeError(f"Failed to process GEBCO bathymetry: {e}")

def load_real_climatology(climatology_path: str) -> Dict[str, xr.Dataset]:
    """Load real NOAA climatology data - FAIL if missing"""
    
    print("📊 Loading real NOAA climatology data...")
    
    climatology_dir = Path(climatology_path)
    if not climatology_dir.exists():
        raise FileNotFoundError(f"Climatology directory not found: {climatology_dir}")
    
    climatology_data = {}
    
    # Required climatology files
    clim_files = {
        'sst': ['era5_grid_sst_climatology.nc', 'noaa_sst_climatology.nc'],
        'slp': ['era5_grid_slp_climatology.nc', 'noaa_slp_climatology.nc']
    }
    
    for var, filenames in clim_files.items():
        loaded = False
        for filename in filenames:
            filepath = climatology_dir / filename
            if filepath.exists():
                file_size = filepath.stat().st_size
                if file_size < 1000:  # Less than 1KB = probably empty
                    print(f"   ⚠️  {filename} is too small ({file_size} bytes), skipping")
                    continue
                
                try:
                    print(f"   📂 Loading {var} from {filename}")
                    ds = xr.open_dataset(filepath)
                    
                    # Validate the dataset
                    if not ds.data_vars:
                        print(f"      ❌ No data variables in {filename}")
                        ds.close()
                        continue
                    
                    print(f"      ✅ Loaded successfully:")
                    print(f"         Variables: {list(ds.data_vars.keys())}")
                    print(f"         Dimensions: {dict(ds.dims)}")
                    print(f"         Size: {file_size / 1e6:.1f} MB")
                    
                    if 'month' in ds.coords:
                        print(f"         Months: {ds.month.values}")
                    
                    climatology_data[var] = ds
                    loaded = True
                    break
                    
                except Exception as e:
                    print(f"      ❌ Error loading {filename}: {e}")
                    continue
        
        if not loaded:
            available_files = [f.name for f in climatology_dir.glob("*.nc")]
            raise FileNotFoundError(
                f"Could not load {var} climatology from any of {filenames}. "
                f"Available files: {available_files}. "
                f"Cannot proceed without real climatology data."
            )
    
    # Check for additional climatology variables we might have
    other_vars = ['u850', 'v850', 'precip', 'z500']
    for var in other_vars:
        possible_files = [f for f in climatology_dir.glob(f"*{var}*.nc")]
        if possible_files:
            try:
                print(f"   📂 Found additional {var} climatology: {possible_files[0].name}")
                ds = xr.open_dataset(possible_files[0])
                climatology_data[var] = ds
                print(f"      ✅ {var} climatology loaded")
            except Exception as e:
                print(f"      ⚠️  Could not load {var} climatology: {e}")
    
    print(f"   ✅ Successfully loaded climatology for {len(climatology_data)} variables")
    return climatology_data

def integrate_era5_file(input_file: str, output_file: str, bathymetry: xr.DataArray, 
                       climatology: Dict[str, xr.Dataset]) -> bool:
    """Integrate climate anomalies and bathymetry into ERA5 file"""
    
    print(f"🔄 Processing {Path(input_file).name}...")
    
    try:
        # Load ERA5 data
        print(f"   📂 Loading ERA5 data...")
        era5_data = xr.open_dataset(input_file)
        
        print(f"      Variables: {len(era5_data.data_vars)}")
        print(f"      Dimensions: {dict(era5_data.dims)}")
        
        # Get month for climatology
        if 'valid_time' in era5_data.coords:
            month = int(era5_data.valid_time.dt.month.values[0])
        elif 'time' in era5_data.coords:
            month = int(era5_data.time.dt.month.values[0])
        else:
            print(f"      ⚠️  No time coordinate found, defaulting to January")
            month = 1
        
        print(f"   📅 Processing month: {month}")
        
        # Compute climate anomalies
        anomalies = compute_climate_anomalies(era5_data, month, climatology)
        
        if not anomalies:
            raise ValueError("No climate anomalies could be computed")
        
        # Add anomalies to dataset
        for var_name, anomaly in anomalies.items():
            era5_data[var_name] = anomaly
            print(f"      ✅ Added {var_name}")
        
        # Interpolate bathymetry to ERA5 grid
        print(f"   🌊 Adding bathymetry...")
        
        # Handle different coordinate names
        if 'lat' in era5_data.coords:
            era5_lat = era5_data.lat.values
            era5_lon = era5_data.lon.values
        elif 'latitude' in era5_data.coords:
            era5_lat = era5_data.latitude.values
            era5_lon = era5_data.longitude.values
        else:
            # Try to find latitude/longitude coordinates
            lat_coords = [coord for coord in era5_data.coords if 'lat' in coord.lower()]
            lon_coords = [coord for coord in era5_data.coords if 'lon' in coord.lower()]
            
            if not lat_coords or not lon_coords:
                raise ValueError(f"Cannot find latitude/longitude coordinates in ERA5 data. Available coords: {list(era5_data.coords.keys())}")
            
            era5_lat = era5_data[lat_coords[0]].values
            era5_lon = era5_data[lon_coords[0]].values
            
            print(f"      Using coordinates: {lat_coords[0]}, {lon_coords[0]}")
        
        print(f"      ERA5 grid: {len(era5_lat)} x {len(era5_lon)}")
        print(f"      Bathymetry grid: {bathymetry.sizes['lat']} x {bathymetry.sizes['lon']}")
        
        bathy_interp = bathymetry.interp(
            lat=era5_lat,
            lon=era5_lon,
            method='linear'
        )
        
        era5_data['ocean_depth'] = bathy_interp
        print(f"      ✅ Bathymetry added")
        
        # Count final features
        total_vars = len(era5_data.data_vars)
        anomaly_vars = len([v for v in era5_data.data_vars if 'anomaly' in v])
        
        print(f"   📊 Final dataset:")
        print(f"      Total variables: {total_vars}")
        print(f"      Climate anomalies: {anomaly_vars}")
        print(f"      Bathymetry: {'✅' if 'ocean_depth' in era5_data.data_vars else '❌'}")
        
        # Add metadata
        era5_data.attrs.update({
            'processing_time': datetime.now().isoformat(),
            'climate_anomalies_added': True,
            'bathymetry_added': True,
            'target_features': 18,
            'feature_description': 'ERA5 base variables + climate anomalies + GEBCO bathymetry',
            'climate_anomalies': list(anomalies.keys()),
            'processing_month': month
        })
        
        # Save enhanced dataset
        print(f"   💾 Saving enhanced dataset...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        era5_data.to_netcdf(output_file)
        era5_data.close()
        
        print(f"   ✅ Enhanced dataset saved: {Path(output_file).name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing {Path(input_file).name}: {e}")
        return False

def main():
    """Main data integration pipeline - REAL DATA ONLY"""
    
    print("🚀 DATA INTEGRATION PIPELINE - V1 TRAINING PREPARATION")
    print("=" * 70)
    print("Goal: Add REAL climate anomalies + bathymetry to ERA5 data")
    print("Target: 18 features (15 base + 6 climate + 1 bathymetry)")
    print("Policy: FAIL if real climatology or bathymetry data missing")
    
    # Configuration
    input_file = Path("data/v1_global/processed/v1_era5_202101.nc")
    output_dir = Path("data/processed_v1") 
    gebco_dir = Path("data/gebco")
    climatology_dir = Path("data/v1_global/climatology")
    
    try:
        # Verify input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"Input ERA5 file not found: {input_file}")
        
        print(f"\n✅ Input file verified: {input_file}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process GEBCO bathymetry (FAIL if missing)
        print(f"\n🌊 STEP 1: Processing GEBCO Bathymetry")
        bathymetry = process_gebco_bathymetry(str(gebco_dir), target_resolution=0.5)
        
        # Step 2: Load real climatology (FAIL if missing)
        print(f"\n📊 STEP 2: Loading Real NOAA Climatology")
        climatology = load_real_climatology(str(climatology_dir))
        
        # Step 3: Integrate features
        print(f"\n🔄 STEP 3: Integrating Climate Features")
        output_file = output_dir / f"enhanced_{input_file.name}"
        
        success = integrate_era5_file(str(input_file), str(output_file), bathymetry, climatology)
        
        if not success:
            raise RuntimeError("Feature integration failed")
        
        # Final verification
        print(f"\n✅ VERIFICATION:")
        with xr.open_dataset(output_file) as ds:
            total_vars = len(ds.data_vars)
            anomaly_vars = len([v for v in ds.data_vars if 'anomaly' in v])
            has_bathymetry = 'ocean_depth' in ds.data_vars
            
            print(f"   Total variables: {total_vars}")
            print(f"   Climate anomalies: {anomaly_vars}")
            print(f"   Bathymetry: {'✅' if has_bathymetry else '❌'}")
            
            if anomaly_vars < 2:
                raise ValueError(f"Insufficient climate anomalies: {anomaly_vars} < 2")
            
            if not has_bathymetry:
                raise ValueError("Bathymetry missing from final dataset")
        
        print(f"\n🎉 DATA INTEGRATION COMPLETE")
        print(f"   ✅ Enhanced file: {output_file}")
        print(f"   ✅ Ready for V1 training with REAL climate teleconnections!")
        
    except Exception as e:
        print(f"\n❌ DATA INTEGRATION FAILED: {e}")
        print(f"   Cannot proceed without real climatology and bathymetry data")
        raise

if __name__ == "__main__":
    main()