# experiments/download_v1_global_data.py
"""
V1 Global Climate-Aware Wave Model Data Download
Complete data pipeline for V1 model training with climate anomaly features
"""

import cdsapi
import os
import numpy as np
import xarray as xr
import requests
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import time
from datetime import datetime
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

class V1GlobalDataDownloader:
    """Download and prepare V1 global training data with climate anomalies"""
    
    def __init__(self, data_root: str = "data/v1_global"):
        self.data_root = Path(data_root)
        self.era5_root = self.data_root / "era5"
        self.climatology_root = self.data_root / "climatology" 
        self.processed_root = self.data_root / "processed"
        
        # Create directories
        for path in [self.era5_root, self.climatology_root, self.processed_root]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize CDS client
        try:
            self.client = cdsapi.Client()
            print("✅ CDS API client initialized")
        except Exception as e:
            print(f"❌ CDS API client failed: {e}")
            print("   Make sure you have ~/.cdsapirc configured")
    
    def download_v1_training_data(self, year: int, months: List[int]):
        """Download V1 training data for specified months"""
        
        print(f"📥 V1 GLOBAL DATA DOWNLOAD: {year}")
        print(f"   Months: {months}")
        print(f"   Target: Climate-aware global wave forecasting")
        print("=" * 60)
        
        for month in months:
            print(f"\n📅 Processing {year}-{month:02d}...")
            
            try:
                # Step 1: Download single-level data (waves + surface atmospheric)
                self._download_single_level_v1(year, month)
                
                # Step 2: Download pressure-level data (upper-air for climate features)
                self._download_pressure_levels_v1(year, month)
                
                # Step 3: Combine into V1 format
                self._combine_v1_month_data(year, month)
                
                print(f"✅ {year}-{month:02d} complete")
                
            except Exception as e:
                print(f"❌ Failed {year}-{month:02d}: {e}")
                if "queue" in str(e).lower():
                    print("⏳ Request queued, waiting...")
                    time.sleep(300)
                    continue
                else:
                    raise
    
    def _download_single_level_v1(self, year: int, month: int):
        """Download V1 single-level variables (waves + surface atmospheric)"""
        
        output_file = self.era5_root / f"era5_single_v1_{year}{month:02d}.nc"
        
        if output_file.exists():
            print(f"   ✅ Single-level already exists: {output_file.name}")
            return
        
        print(f"   📥 Downloading single-level variables...")
        
        # V1 single-level variables (per schema)
        variables = [
            # Wave variables (4 for V1)
            'significant_height_of_combined_wind_waves_and_swell',  # swh
            'mean_wave_direction',                                  # mwd
            'mean_wave_period',                                     # mwp  
            'significant_height_of_wind_waves',                     # shww
            
            # Surface atmospheric variables (5 for V1)
            '10m_u_component_of_wind',                             # u10
            '10m_v_component_of_wind',                             # v10
            'mean_sea_level_pressure',                             # msl
            'sea_surface_temperature',                             # sst
            'total_precipitation'                                  # tp
        ]
        
        print(f"      Variables: {len(variables)}")
        for var in variables:
            print(f"        • {var}")
        
        self.client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],  # 6-hourly for V1
            'area': [90, -180, -90, 179.75],  # Global domain
            'format': 'netcdf',
        }, str(output_file))
        
        print(f"   ✅ Single-level download complete")
    
    def _download_pressure_levels_v1(self, year: int, month: int):
        """Download V1 pressure-level variables (for climate features)"""
        
        output_file = self.era5_root / f"era5_pressure_v1_{year}{month:02d}.nc"
        
        if output_file.exists():
            print(f"   ✅ Pressure-level already exists: {output_file.name}")
            return
        
        print(f"   📥 Downloading pressure-level variables...")
        
        # V1 pressure-level variables (per schema)
        variables = [
            'u_component_of_wind',    # u850, u500
            'v_component_of_wind',    # v850, v500  
            'geopotential'            # z850, z500
        ]
        
        pressure_levels = [850, 500]  # V1 uses 850 and 500 hPa
        
        print(f"      Variables: {variables}")
        print(f"      Levels: {pressure_levels} hPa")
        
        self.client.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'variable': variables,
            'pressure_level': pressure_levels,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],  # 6-hourly
            'area': [90, -180, -90, 179.75],  # Global domain
            'format': 'netcdf',
        }, str(output_file))
        
        print(f"   ✅ Pressure-level download complete")
    
    def _combine_v1_month_data(self, year: int, month: int):
        """Combine single-level and pressure-level into V1 format"""
        
        output_file = self.processed_root / f"v1_era5_{year}{month:02d}.nc"
        
        if output_file.exists():
            print(f"   ✅ Combined V1 data already exists: {output_file.name}")
            return
        
        print(f"   🔗 Combining V1 training data...")
        
        # Load datasets
        single_file = self.era5_root / f"era5_single_v1_{year}{month:02d}.nc"
        pressure_file = self.era5_root / f"era5_pressure_v1_{year}{month:02d}.nc"
        
        if not single_file.exists():
            print(f"   ❌ Missing single-level file: {single_file}")
            return
            
        if not pressure_file.exists():
            print(f"   ❌ Missing pressure-level file: {pressure_file}")
            return
        
        try:
            # Load datasets
            ds_single = xr.open_dataset(single_file)
            ds_pressure = xr.open_dataset(pressure_file)
            
            print(f"      Single-level shape: {dict(ds_single.dims)}")
            print(f"      Pressure-level shape: {dict(ds_pressure.dims)}")
            
            # Start with single-level data
            combined_vars = {}
            
            # Add single-level variables with V1 naming
            var_mapping = {
                'significant_height_of_combined_wind_waves_and_swell': 'swh',
                'mean_wave_direction': 'mwd',
                'mean_wave_period': 'mwp',
                'significant_height_of_wind_waves': 'shww',
                '10m_u_component_of_wind': 'u10',
                '10m_v_component_of_wind': 'v10',
                'mean_sea_level_pressure': 'msl',
                'sea_surface_temperature': 'sst',
                'total_precipitation': 'tp'
            }
            
            for era5_name, v1_name in var_mapping.items():
                if era5_name in ds_single.data_vars:
                    combined_vars[v1_name] = ds_single[era5_name]
                    print(f"      ✅ {v1_name}")
                else:
                    print(f"      ❌ Missing: {era5_name}")
            
            # Add pressure-level variables
            pressure_var_mapping = {
                'u_component_of_wind': 'u',
                'v_component_of_wind': 'v', 
                'geopotential': 'z'
            }
            
            for era5_name, base_name in pressure_var_mapping.items():
                if era5_name in ds_pressure.data_vars:
                    var_data = ds_pressure[era5_name]
                    
                    # Extract specific levels for V1
                    for level in [850, 500]:
                        try:
                            if 'level' in var_data.coords:
                                level_data = var_data.sel(level=level)
                            elif 'pressure_level' in var_data.coords:
                                level_data = var_data.sel(pressure_level=level)
                            else:
                                print(f"      ❌ No pressure coordinate found for {era5_name}")
                                continue
                            
                            v1_var_name = f"{base_name}{level}"
                            combined_vars[v1_var_name] = level_data
                            print(f"      ✅ {v1_var_name}")
                            
                        except Exception as e:
                            print(f"      ❌ Failed to extract {base_name}{level}: {e}")
            
            # Create combined dataset
            combined_ds = xr.Dataset(combined_vars, coords=ds_single.coords)
            
            # Add V1 metadata
            combined_ds.attrs.update({
                'title': f'V1 Global Climate-Aware Training Data {year}-{month:02d}',
                'source': 'ERA5 reanalysis',
                'model_version': 'V1',
                'variables_count': len(combined_vars),
                'features_for_model': '11 base features (4 wave + 5 surface + 2 pressure)',
                'climate_anomalies': 'To be computed from climatology',
                'processing_date': datetime.now().isoformat(),
                'spatial_resolution': '0.25 degrees',
                'temporal_resolution': '6 hours'
            })
            
            print(f"      Total variables: {len(combined_vars)}")
            
            # Save with compression
            encoding = {var: {'zlib': True, 'complevel': 1} for var in combined_vars}
            combined_ds.to_netcdf(output_file, encoding=encoding)
            
            # Cleanup
            ds_single.close()
            ds_pressure.close()
            combined_ds.close()
            
            # Quick verification
            with xr.open_dataset(output_file) as verify_ds:
                print(f"   ✅ V1 data saved: {output_file.name}")
                print(f"      File size: {output_file.stat().st_size / (1024**2):.1f} MB")
                print(f"      Variables: {len(verify_ds.data_vars)}")
                print(f"      Time steps: {len(verify_ds.time) if 'time' in verify_ds.dims else len(verify_ds.valid_time)}")
            
        except Exception as e:
            print(f"   ❌ Failed to combine V1 data: {e}")
            import traceback
            traceback.print_exc()
    
    def download_noaa_climatology(self):
        """Download NOAA pre-computed climatologies for V1 climate anomalies"""
        
        print(f"\n🌡️  DOWNLOADING NOAA CLIMATOLOGIES")
        print(f"   Purpose: Climate anomaly computation for V1")
        print(f"   Resolution: 2.5° → will interpolate to 0.25°")
        print("=" * 60)
        
        # NOAA climatology variables for V1
        noaa_vars = {
            'sst': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.mon.ltm.1991-2020.nc',
                'description': 'Sea Surface Temperature climatology'
            },
            'slp': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.mon.ltm.nc',
                'description': 'Sea Level Pressure climatology'
            },
            'uwnd': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/uwnd.mon.ltm.nc',
                'description': 'U-component wind climatology'
            },
            'vwnd': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/vwnd.mon.ltm.nc', 
                'description': 'V-component wind climatology'
            },
            'precip': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.mon.ltm.nc',
                'description': 'Precipitation climatology'
            },
            'hgt': {
                'url': 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/hgt.mon.ltm.nc',
                'description': 'Geopotential height climatology'
            }
        }
        
        climatology_files = {}
        
        for var_name, var_info in noaa_vars.items():
            output_file = self.climatology_root / f"noaa_{var_name}_climatology.nc"
            
            if output_file.exists():
                print(f"   ✅ {var_name} climatology already exists")
                climatology_files[var_name] = output_file
                continue
            
            print(f"   📥 Downloading {var_name} climatology...")
            print(f"      {var_info['description']}")
            
            try:
                # Download NOAA climatology
                response = requests.get(var_info['url'], stream=True)
                response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"   ✅ {var_name} climatology downloaded")
                climatology_files[var_name] = output_file
                
            except Exception as e:
                print(f"   ❌ Failed to download {var_name}: {e}")
                print(f"      URL: {var_info['url']}")
                print(f"      You may need to download manually")
        
        return climatology_files
    
    def interpolate_climatologies_to_era5_grid(self, era5_sample_file: str):
        """Interpolate NOAA climatologies to ERA5 0.25° grid"""
        
        print(f"\n🔄 INTERPOLATING CLIMATOLOGIES TO ERA5 GRID")
        print(f"   Target: 0.25° resolution matching ERA5")
        print("=" * 60)
        
        # Load ERA5 grid as template
        print(f"   📖 Loading ERA5 grid template: {Path(era5_sample_file).name}")
        
        try:
            with xr.open_dataset(era5_sample_file) as era5_sample:
                # Get ERA5 grid coordinates
                if 'latitude' in era5_sample.coords:
                    era5_lats = era5_sample.latitude.values
                    era5_lons = era5_sample.longitude.values
                elif 'lat' in era5_sample.coords:
                    era5_lats = era5_sample.lat.values
                    era5_lons = era5_sample.lon.values
                else:
                    raise ValueError("No latitude/longitude coordinates found")
                
                print(f"      ERA5 grid: {len(era5_lats)} × {len(era5_lons)}")
                print(f"      Lat range: {era5_lats.min():.2f}° to {era5_lats.max():.2f}°")
                print(f"      Lon range: {era5_lons.min():.2f}° to {era5_lons.max():.2f}°")
        
        except Exception as e:
            print(f"   ❌ Failed to load ERA5 template: {e}")
            return {}
        
        # Interpolate each climatology
        interpolated_files = {}
        
        climatology_files = list(self.climatology_root.glob("noaa_*_climatology.nc"))
        
        for clim_file in climatology_files:
            var_name = clim_file.stem.replace('noaa_', '').replace('_climatology', '')
            output_file = self.climatology_root / f"era5_grid_{var_name}_climatology.nc"
            
            if output_file.exists():
                print(f"   ✅ {var_name} already interpolated")
                interpolated_files[var_name] = output_file
                continue
            
            print(f"   🔄 Interpolating {var_name} climatology...")
            
            try:
                # Load NOAA climatology
                with xr.open_dataset(clim_file) as noaa_clim:
                    
                    # Get NOAA grid
                    if 'lat' in noaa_clim.coords:
                        noaa_lats = noaa_clim.lat.values
                        noaa_lons = noaa_clim.lon.values
                    elif 'latitude' in noaa_clim.coords:
                        noaa_lats = noaa_clim.latitude.values
                        noaa_lons = noaa_clim.longitude.values
                    else:
                        print(f"      ❌ No coordinates found in {clim_file}")
                        continue
                    
                    print(f"      NOAA grid: {len(noaa_lats)} × {len(noaa_lons)}")
                    
                    # Find the main climatology variable
                    clim_var = None
                    for var in noaa_clim.data_vars:
                        if len(noaa_clim[var].dims) >= 3:  # Time, lat, lon
                            clim_var = var
                            break
                    
                    if clim_var is None:
                        print(f"      ❌ No suitable variable found in {clim_file}")
                        continue
                    
                    print(f"      Climatology variable: {clim_var}")
                    clim_data = noaa_clim[clim_var]
                    
                    # Handle different coordinate names and ensure proper longitude range
                    if noaa_lons.max() > 180:
                        # Convert 0-360 to -180-180
                        noaa_lons = np.where(noaa_lons > 180, noaa_lons - 360, noaa_lons)
                    
                    # Create meshgrids
                    noaa_lon_grid, noaa_lat_grid = np.meshgrid(noaa_lons, noaa_lats)
                    era5_lon_grid, era5_lat_grid = np.meshgrid(era5_lons, era5_lats)
                    
                    # Interpolate for each month
                    interpolated_months = []
                    
                    for month in range(12):
                        print(f"      Month {month+1:2d}/12...", end=' ')
                        
                        # Get monthly data
                        monthly_data = clim_data.isel(time=month).values
                        
                        # Flatten grids for interpolation
                        points = np.column_stack([noaa_lon_grid.ravel(), noaa_lat_grid.ravel()])
                        values = monthly_data.ravel()
                        
                        # Remove NaN values
                        valid_mask = np.isfinite(values)
                        points = points[valid_mask]
                        values = values[valid_mask]
                        
                        if len(values) == 0:
                            print(f"No valid data!")
                            interpolated_months.append(np.full_like(era5_lat_grid, np.nan))
                            continue
                        
                        # Interpolate to ERA5 grid
                        era5_points = np.column_stack([era5_lon_grid.ravel(), era5_lat_grid.ravel()])
                        
                        interpolated_values = griddata(
                            points, values, era5_points, 
                            method='linear', fill_value=np.nan
                        )
                        
                        interpolated_monthly = interpolated_values.reshape(era5_lat_grid.shape)
                        interpolated_months.append(interpolated_monthly)
                        
                        print(f"✅")
                    
                    # Create interpolated dataset
                    interpolated_data = np.array(interpolated_months)
                    
                    # Create new dataset
                    interpolated_ds = xr.Dataset({
                        f'{var_name}_climatology': (['time', 'latitude', 'longitude'], interpolated_data)
                    }, coords={
                        'time': range(1, 13),  # Months 1-12
                        'latitude': era5_lats,
                        'longitude': era5_lons
                    })
                    
                    # Add metadata
                    interpolated_ds.attrs.update({
                        'title': f'{var_name.upper()} climatology interpolated to ERA5 grid',
                        'source': f'NOAA climatology interpolated from {clim_file.name}',
                        'original_resolution': f'{len(noaa_lats)}x{len(noaa_lons)}',
                        'target_resolution': f'{len(era5_lats)}x{len(era5_lons)}',
                        'interpolation_method': 'linear',
                        'interpolation_date': datetime.now().isoformat()
                    })
                    
                    # Save interpolated climatology
                    interpolated_ds.to_netcdf(output_file)
                    interpolated_ds.close()
                    
                    print(f"      ✅ Saved: {output_file.name}")
                    interpolated_files[var_name] = output_file
            
            except Exception as e:
                print(f"      ❌ Failed to interpolate {var_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n   📊 Interpolation summary:")
        print(f"      Successful: {len(interpolated_files)}")
        print(f"      Required for V1: 6 (sst, slp, uwnd, vwnd, precip, hgt)")
        
        return interpolated_files
    
    def compute_v1_climate_anomalies(self, year: int, month: int):
        """Compute climate anomalies for V1 training data"""
        
        print(f"\n🌡️  COMPUTING V1 CLIMATE ANOMALIES: {year}-{month:02d}")
        
        # Load V1 training data
        v1_file = self.processed_root / f"v1_era5_{year}{month:02d}.nc"
        output_file = self.processed_root / f"v1_era5_with_anomalies_{year}{month:02d}.nc"
        
        if output_file.exists():
            print(f"   ✅ Anomalies already computed: {output_file.name}")
            return output_file
        
        if not v1_file.exists():
            print(f"   ❌ V1 training data not found: {v1_file}")
            return None
        
        print(f"   📖 Loading V1 data...")
        
        try:
            with xr.open_dataset(v1_file) as v1_ds:
                
                # Find interpolated climatologies
                clim_files = list(self.climatology_root.glob("era5_grid_*_climatology.nc"))
                
                if len(clim_files) == 0:
                    print(f"   ❌ No interpolated climatologies found")
                    print(f"      Run interpolate_climatologies_to_era5_grid() first")
                    return None
                
                print(f"   📖 Loading climatologies...")
                climatologies = {}
                
                for clim_file in clim_files:
                    var_name = clim_file.stem.replace('era5_grid_', '').replace('_climatology', '')
                    
                    with xr.open_dataset(clim_file) as clim_ds:
                        clim_var_name = f'{var_name}_climatology'
                        if clim_var_name in clim_ds.data_vars:
                            # Get climatology for this month
                            monthly_clim = clim_ds[clim_var_name].sel(time=month)
                            climatologies[var_name] = monthly_clim.values
                            print(f"      ✅ {var_name} climatology loaded")
                        else:
                            print(f"      ❌ {clim_var_name} not found in {clim_file}")
                
                print(f"   🧮 Computing anomalies...")
                
                # Compute V1 climate anomalies (6 features)
                anomaly_vars = {}
                
                # V1 anomaly mappings (per schema)
                anomaly_mappings = {
                    'sst_anomaly': ('sst', 'sst'),
                    'slp_anomaly': ('msl', 'slp'), 
                    'u850_anomaly': ('u850', 'uwnd'),
                    'v850_anomaly': ('v850', 'vwnd'),
                    'precip_anomaly': ('tp', 'precip'),
                    'z500_anomaly': ('z500', 'hgt')
                }
                
                for anomaly_name, (v1_var, clim_var) in anomaly_mappings.items():
                    if v1_var in v1_ds.data_vars and clim_var in climatologies:
                        
                        current_data = v1_ds[v1_var].values
                        climatology = climatologies[clim_var]
                        
                        # Compute anomaly
                        anomaly = current_data - climatology[np.newaxis, :, :]  # Broadcast over time
                        
                        # Create DataArray
                        anomaly_da = xr.DataArray(
                            anomaly,
                            dims=v1_ds[v1_var].dims,
                            coords=v1_ds[v1_var].coords,
                            attrs={
                                'long_name': f'{v1_var} anomaly relative to climatology',
                                'units': v1_ds[v1_var].attrs.get('units', ''),
                                'climatology_source': f'{clim_var} climatology',
                                'climatology_period': '1991-2020'
                            }
                        )
                        
                        anomaly_vars[anomaly_name] = anomaly_da
                        print(f"      ✅ {anomaly_name}")
                        
                    else:
                        print(f"      ❌ Missing data for {anomaly_name}: {v1_var}, {clim_var}")
                
                # Create enhanced V1 dataset with anomalies
                enhanced_vars = dict(v1_ds.data_vars)
                enhanced_vars.update(anomaly_vars)
                
                enhanced_ds = xr.Dataset(enhanced_vars, coords=v1_ds.coords)
                
                # Update metadata
                enhanced_ds.attrs.update(v1_ds.attrs)
                enhanced_ds.attrs.update({
                    'climate_anomalies_computed': True,
                    'climate_anomaly_count': len(anomaly_vars),
                    'total_features_v1': f"17 (11 base + {len(anomaly_vars)} climate anomalies)",
                    'anomaly_computation_date': datetime.now().isoformat(),
                    'climatology_source': 'NOAA interpolated to ERA5 grid'
                })
                
                print(f"   💾 Saving enhanced V1 data...")
                
                # Save with compression
                encoding = {var: {'zlib': True, 'complevel': 1} for var in enhanced_vars}
                enhanced_ds.to_netcdf(output_file, encoding=encoding)
                
                enhanced_ds.close()
                
                print(f"   ✅ V1 with anomalies saved: {output_file.name}")
                print(f"      Total variables: {len(enhanced_vars)}")
                print(f"      File size: {output_file.stat().st_size / (1024**2):.1f} MB")
                
                return output_file
        
        except Exception as e:
            print(f"   ❌ Failed to compute anomalies: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_v1_training_dataset(self, year: int, months: List[int], era5_sample_file: str = None):
        """Complete V1 training dataset preparation pipeline"""
        
        print(f"🚀 V1 GLOBAL TRAINING DATASET PREPARATION")
        print(f"   Year: {year}")  
        print(f"   Months: {months}")
        print(f"   Target: 17-feature climate-aware global wave model")
        print("=" * 70)
        
        # Step 1: Download V1 training data
        print(f"\n📥 STEP 1: Download V1 ERA5 data")
        self.download_v1_training_data(year, months)
        
        # Step 2: Download NOAA climatologies
        print(f"\n🌡️  STEP 2: Download NOAA climatologies")
        climatology_files = self.download_noaa_climatology()
        
        # Step 3: Interpolate climatologies to ERA5 grid
        print(f"\n🔄 STEP 3: Interpolate climatologies to ERA5 grid")
        
        # Use first month as template if no sample file provided
        if era5_sample_file is None:
            era5_sample_file = self.processed_root / f"v1_era5_{year}{months[0]:02d}.nc"
        
        if not Path(era5_sample_file).exists():
            print(f"   ❌ ERA5 sample file not found: {era5_sample_file}")
            print(f"      Make sure V1 data download completed successfully")
            return False
        
        interpolated_files = self.interpolate_climatologies_to_era5_grid(str(era5_sample_file))
        
        if len(interpolated_files) < 6:
            print(f"   ⚠️  Only {len(interpolated_files)}/6 climatologies interpolated")
            print(f"      V1 model requires 6 climate anomaly features")
        
        # Step 4: Compute climate anomalies for each month
        print(f"\n🧮 STEP 4: Compute climate anomalies")
        
        enhanced_files = []
        for month in months:
            anomaly_file = self.compute_v1_climate_anomalies(year, month)
            if anomaly_file:
                enhanced_files.append(anomaly_file)
        
        # Step 5: Validation and summary
        print(f"\n✅ STEP 5: V1 dataset validation")
        
        total_size_mb = 0
        total_timesteps = 0
        
        for enhanced_file in enhanced_files:
            if enhanced_file and enhanced_file.exists():
                file_size_mb = enhanced_file.stat().st_size / (1024**2)
                total_size_mb += file_size_mb
                
                # Quick validation
                with xr.open_dataset(enhanced_file) as ds:
                    time_coord = 'time' if 'time' in ds.dims else 'valid_time'
                    timesteps = len(ds[time_coord])
                    total_timesteps += timesteps
                    
                    # Check for V1 required variables
                    required_vars = [
                        'swh', 'mwd', 'mwp', 'shww',  # Wave variables
                        'u10', 'v10', 'msl', 'sst', 'tp',  # Surface atmospheric
                        'u850', 'v850', 'z500',  # Pressure level (subset)
                        'sst_anomaly', 'slp_anomaly', 'u850_anomaly',  # Climate anomalies
                        'v850_anomaly', 'precip_anomaly', 'z500_anomaly'
                    ]
                    
                    missing_vars = [var for var in required_vars if var not in ds.data_vars]
                    
                    if missing_vars:
                        print(f"   ⚠️  {enhanced_file.name}: Missing {len(missing_vars)} variables")
                        for var in missing_vars:
                            print(f"        • {var}")
                    else:
                        print(f"   ✅ {enhanced_file.name}: All 17 V1 features present")
                        print(f"      File size: {file_size_mb:.1f} MB, Timesteps: {timesteps}")
        
        print(f"\n🎯 V1 TRAINING DATASET READY!")
        print(f"   📁 Location: {self.processed_root}")
        print(f"   📊 Files: {len(enhanced_files)} months")
        print(f"   📏 Total size: {total_size_mb:.1f} MB")
        print(f"   ⏰ Total timesteps: {total_timesteps}")
        print(f"   🧠 Features: 17 (11 base + 6 climate anomalies)")
        print(f"   🎲 Ready for: Global climate-aware wave model training")
        
        # Generate training configuration summary
        self._generate_v1_training_config(enhanced_files, year, months)
        
        return True
    
    def _generate_v1_training_config(self, data_files: List[Path], year: int, months: List[int]):
        """Generate V1 training configuration file"""
        
        config_file = self.data_root / "v1_training_config.json"
        
        config = {
            "model_version": "V1",
            "description": "Global Climate-Aware Variable LR Wave Model",
            "data_preparation": {
                "year": year,
                "months": months,
                "data_files": [str(f.relative_to(self.data_root)) for f in data_files if f and f.exists()],
                "total_files": len([f for f in data_files if f and f.exists()]),
                "preparation_date": datetime.now().isoformat()
            },
            "model_architecture": {
                "input_features": 17,
                "base_features": 11,
                "climate_anomalies": 6,
                "output_features": 4,
                "feature_list": {
                    "wave_variables": ["swh", "mwd", "mwp", "shww"],
                    "surface_atmospheric": ["u10", "v10", "msl", "sst", "tp"],
                    "pressure_levels": ["u850", "v850", "z500"],
                    "climate_anomalies": ["sst_anomaly", "slp_anomaly", "u850_anomaly", 
                                        "v850_anomaly", "precip_anomaly", "z500_anomaly"]
                },
                "target_outputs": ["swh", "mwd_cos", "mwd_sin", "mwp"]
            },
            "training_parameters": {
                "sequence_length": 6,
                "prediction_horizon": 4,
                "batch_size": 6,
                "base_learning_rate": 1e-4,
                "variable_lr_multipliers": {
                    "swh": 0.7,
                    "mwd": 1.0,
                    "mwp": 1.3
                },
                "early_stopping_patience": 50,
                "max_epochs": 200
            },
            "benchmark_targets": {
                "primary": "Beat EarthFormer 0.22m RMSE",
                "secondary": "Seasonal consistency (no summer collapse)",
                "innovation": "Climate teleconnection capture"
            },
            "data_sources": {
                "training_data": "ERA5 reanalysis",
                "climatology": "NOAA pre-computed, interpolated to ERA5 grid",
                "resolution": "0.25 degrees",
                "temporal_frequency": "6 hours"
            }
        }
        
        with open(config_file, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        print(f"   📋 Training config saved: {config_file.name}")
    
    def quick_data_inspection(self, file_path: str):
        """Quick inspection of V1 training data quality"""
        
        print(f"\n🔍 QUICK V1 DATA INSPECTION: {Path(file_path).name}")
        
        try:
            with xr.open_dataset(file_path) as ds:
                print(f"   📊 Dataset overview:")
                print(f"      Variables: {len(ds.data_vars)}")
                print(f"      Dimensions: {dict(ds.dims)}")
                
                time_coord = 'time' if 'time' in ds.dims else 'valid_time'
                print(f"      Time steps: {len(ds[time_coord])}")
                
                # Check data quality for key variables
                key_vars = ['swh', 'mwd', 'mwp', 'sst_anomaly', 'slp_anomaly']
                
                print(f"\n   🌊 Data quality check:")
                for var in key_vars:
                    if var in ds.data_vars:
                        values = ds[var].values
                        nan_percent = (np.isnan(values).sum() / values.size) * 100
                        
                        if np.any(np.isfinite(values)):
                            finite_vals = values[np.isfinite(values)]
                            val_range = f"[{np.min(finite_vals):.3f}, {np.max(finite_vals):.3f}]"
                        else:
                            val_range = "All NaN"
                        
                        print(f"      {var:>12}: {nan_percent:>5.1f}% NaN, range {val_range}")
                    else:
                        print(f"      {var:>12}: ❌ Missing")
                
                # Wave data context
                if any(var in ds.data_vars for var in ['swh', 'mwd', 'mwp']):
                    wave_vars = [var for var in ['swh', 'mwd', 'mwp'] if var in ds.data_vars]
                    avg_nan_percent = np.mean([
                        (np.isnan(ds[var].values).sum() / ds[var].values.size) * 100 
                        for var in wave_vars
                    ])
                    
                    print(f"\n   🌊 Wave data assessment:")
                    if avg_nan_percent <= 50:
                        print(f"      ✅ Quality: GOOD ({avg_nan_percent:.1f}% NaN - normal for ocean data)")
                    elif avg_nan_percent <= 70:
                        print(f"      ⚠️  Quality: MODERATE ({avg_nan_percent:.1f}% NaN)")
                    else:
                        print(f"      ❌ Quality: POOR ({avg_nan_percent:.1f}% NaN - investigate)")
        
        except Exception as e:
            print(f"   ❌ Inspection failed: {e}")


def main():
    """Main V1 data preparation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare V1 global training data')
    parser.add_argument('--year', type=int, default=2021, help='Year to download')
    parser.add_argument('--months', nargs='+', type=int, default=[1, 2], 
                       help='Months to download (e.g., --months 1 2 for Jan-Feb)')
    parser.add_argument('--data_root', default='data/v1_global', help='Data root directory')
    parser.add_argument('--climatology_only', action='store_true', 
                       help='Only download and process climatologies')
    parser.add_argument('--inspect', type=str, help='Inspect specific data file')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = V1GlobalDataDownloader(args.data_root)
    
    if args.inspect:
        downloader.quick_data_inspection(args.inspect)
        return
    
    if args.climatology_only:
        print("🌡️  CLIMATOLOGY-ONLY MODE")
        climatology_files = downloader.download_noaa_climatology()
        
        # Need a sample ERA5 file for grid template
        sample_files = list(downloader.processed_root.glob("v1_era5_*.nc"))
        if sample_files:
            interpolated_files = downloader.interpolate_climatologies_to_era5_grid(str(sample_files[0]))
            print(f"✅ Climatology preparation complete: {len(interpolated_files)} variables")
        else:
            print("❌ No ERA5 sample files found for grid interpolation")
        return
    
    # Full V1 preparation pipeline
    print(f"🚀 V1 GLOBAL DATA PREPARATION")
    print(f"   Year: {args.year}")
    print(f"   Months: {args.months}")
    print(f"   Data root: {args.data_root}")
    print(f"   Target: Climate-aware global wave forecasting model")
    
    # Prepare complete V1 training dataset
    success = downloader.prepare_v1_training_dataset(args.year, args.months)
    
    if success:
        print(f"\n🎉 V1 DATA PREPARATION COMPLETE!")
        print(f"   Ready for global climate-aware wave model training")
        print(f"   Next step: python train_v1_global_model.py")
        
        # Quick inspection of first file
        data_files = list(downloader.processed_root.glob("v1_era5_with_anomalies_*.nc"))
        if data_files:
            downloader.quick_data_inspection(str(data_files[0]))
    else:
        print(f"\n❌ V1 DATA PREPARATION FAILED")
        print(f"   Check error messages above for issues")


if __name__ == "__main__":
    main()