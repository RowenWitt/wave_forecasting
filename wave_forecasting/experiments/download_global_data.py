# experiments/download_global_data.py
"""Global data download utilities for climate-aware wave forecasting"""
import cdsapi
import os
from typing import List, Optional
import time
import xarray as xr

from config.base import GlobalDataConfig

class GlobalERA5Downloader:
    """Download global ERA5 data for climate-aware wave forecasting"""
    
    def __init__(self, config: GlobalDataConfig):
        self.config = config
        self.client = cdsapi.Client()
        os.makedirs(config.era5_root, exist_ok=True)
        os.makedirs(config.climatology_root, exist_ok=True)
    
    def download_year_data(self, year: int, months: List[int] = None):
        """Download a full year of global ERA5 data"""
        
        if months is None:
            months = list(range(1, 13))
        
        for month in months:
            print(f"📥 Downloading global {year}-{month:02d}...")
            
            try:
                # Download single-level data (surface + waves)
                self._download_month_single_level(year, month)
                
                # Download pressure-level data (upper-air)
                self._download_month_pressure_levels(year, month)
                
                print(f"✅ Global {year}-{month:02d} complete")
                
            except Exception as e:
                print(f"❌ Failed {year}-{month:02d}: {e}")
                if "queue" in str(e).lower() or "request" in str(e).lower():
                    print("⏳ Waiting for queue...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
    
    def _download_month_single_level(self, year: int, month: int):
        """Download single-level data (surface + waves) for one month"""
        
        output_file = f"{self.config.era5_root}/era5_single_level_{year}{month:02d}.nc"
        
        if os.path.exists(output_file):
            print(f"   Single-level data already exists: {output_file}")
            return
        
        print(f"   Downloading single-level data...")
        
        # Get all single-level variables (atmospheric + waves)
        variables = self.config.get_single_level_download_vars()
        
        self.client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, self.config.time_step_hours)],
            'area': [self.config.lat_bounds[1], self.config.lon_bounds[0], 
                    self.config.lat_bounds[0], self.config.lon_bounds[1]],
            'format': 'netcdf',
        }, output_file)
    
    def _download_month_pressure_levels(self, year: int, month: int):
        """Download pressure-level data for one month"""
        
        output_file = f"{self.config.era5_root}/era5_pressure_levels_{year}{month:02d}.nc"
        
        if os.path.exists(output_file):
            print(f"   Pressure-level data already exists: {output_file}")
            return
        
        print(f"   Downloading pressure-level data...")
        
        # Get pressure-level variables
        variables = self.config.get_pressure_level_download_vars()
        
        self.client.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'variable': variables,
            'pressure_level': self.config.pressure_levels,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, self.config.time_step_hours)],
            'area': [self.config.lat_bounds[1], self.config.lon_bounds[0], 
                    self.config.lat_bounds[0], self.config.lon_bounds[1]],
            'format': 'netcdf',
        }, output_file)
    
    def download_multi_year(self, start_year: int, end_year: int):
        """Download multiple years of global data"""
        
        print(f"🌍 Starting global ERA5 download: {start_year}-{end_year}")
        print(f"   Variables: {len(self.config.get_single_level_download_vars())} single-level")
        print(f"   Variables: {len(self.config.get_pressure_level_download_vars())} pressure-level")
        print(f"   Pressure levels: {self.config.pressure_levels}")
        print(f"   Domain: {self.config.lat_bounds}°N, {self.config.lon_bounds}°E")
        
        for year in range(start_year, end_year + 1):
            print(f"\n🗓️  Downloading year {year}...")
            self.download_year_data(year)
            print(f"✅ Year {year} complete!")
    
    def download_climatology_data(self):
        """Download climatology data for anomaly computation"""
        
        start_year, end_year = self.config.climatology_years
        
        print(f"🌡️  Downloading climatology data: {start_year}-{end_year}")
        print("   This will be used to compute climate anomalies")
        
        for year in range(start_year, end_year + 1):
            print(f"   Climatology year {year}...")
            self.download_year_data(year)
    
    def combine_and_process_month(self, year: int, month: int, output_file: Optional[str] = None):
        """Combine single-level and pressure-level data for a month"""
        
        if output_file is None:
            output_file = f"{self.config.processed_root}/era5_combined_{year}{month:02d}.nc"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if os.path.exists(output_file):
            print(f"   Combined data already exists: {output_file}")
            return
        
        print(f"   Combining data for {year}-{month:02d}...")
        
        # Load single-level data
        single_level_file = f"{self.config.era5_root}/era5_single_level_{year}{month:02d}.nc"
        pressure_level_file = f"{self.config.era5_root}/era5_pressure_levels_{year}{month:02d}.nc"
        
        if not os.path.exists(single_level_file):
            print(f"   Missing single-level file: {single_level_file}")
            return
        
        if not os.path.exists(pressure_level_file):
            print(f"   Missing pressure-level file: {pressure_level_file}")
            return
        
        try:
            # Load datasets
            ds_single = xr.open_dataset(single_level_file)
            ds_pressure = xr.open_dataset(pressure_level_file)
            
            # Extract specific pressure levels and rename
            combined_vars = {}
            
            # Add single-level variables
            for var in ds_single.data_vars:
                combined_vars[var] = ds_single[var]
            
            # Add pressure-level variables at specific levels
            for var in self.config.pressure_level_vars:
                if var in ds_pressure.data_vars:
                    for level in self.config.pressure_levels:
                        try:
                            level_data = ds_pressure[var].sel(level=level)
                            new_var_name = f"{var}_{level}"
                            combined_vars[new_var_name] = level_data
                        except KeyError:
                            print(f"   Warning: Level {level} not found for {var}")
            
            # Create combined dataset
            combined_ds = xr.Dataset(combined_vars, coords=ds_single.coords)
            
            # Add metadata
            combined_ds.attrs.update({
                'title': f'Combined ERA5 data for {year}-{month:02d}',
                'source': 'ERA5 reanalysis',
                'processing': 'Combined single-level and pressure-level data',
                'pressure_levels': str(self.config.pressure_levels)
            })
            
            # Save combined dataset
            print(f"   Saving combined dataset...")
            combined_ds.to_netcdf(output_file)
            
            # Close datasets
            ds_single.close()
            ds_pressure.close()
            combined_ds.close()
            
            print(f"   ✅ Combined data saved: {output_file}")
            
        except Exception as e:
            print(f"   ❌ Failed to combine data: {e}")
    
    def process_year_data(self, year: int):
        """Process all months for a year"""
        
        print(f"🔄 Processing year {year} data...")
        
        for month in range(1, 13):
            self.combine_and_process_month(year, month)

def download_global_gebco_data(config: GlobalDataConfig):
    """Download global GEBCO bathymetry data"""
    
    import requests
    
    os.makedirs(config.gebco_root, exist_ok=True)
    output_file = f"{config.gebco_root}/gebco_global.nc"
    
    if os.path.exists(output_file):
        print("🌍 Global GEBCO data already exists")
        return
    
    print("📥 Downloading global GEBCO bathymetry...")
    
    try:
        # GEBCO OPeNDAP server - global dataset
        gebco_url = "https://www.bodc.ac.uk/thredds/dodsC/GEBCO_2023/GEBCO_2023.nc"
        
        print("   Opening global GEBCO dataset...")
        gebco_global = xr.open_dataset(gebco_url)
        
        # Determine coordinate names
        lat_coord = 'lat' if 'lat' in gebco_global.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in gebco_global.coords else 'longitude'
        
        print(f"   Subsetting to global domain: {config.lat_bounds}, {config.lon_bounds}")
        
        # For global data, we want the full range but may need to handle longitude wrapping
        if config.lon_bounds[1] > 180:
            # Handle 0-360 longitude convention
            gebco_subset = gebco_global.sel({
                lat_coord: slice(config.lat_bounds[0], config.lat_bounds[1])
            })
            # Full longitude range
        else:
            # Standard -180 to 180 convention
            gebco_subset = gebco_global.sel({
                lat_coord: slice(config.lat_bounds[0], config.lat_bounds[1]),
                lon_coord: slice(config.lon_bounds[0], config.lon_bounds[1])
            })
        
        print("   Downloading GEBCO subset (this may take a while for global data)...")
        gebco_subset.load()
        
        print("   Saving GEBCO data...")
        gebco_subset.to_netcdf(output_file)
        gebco_subset.close()
        gebco_global.close()
        
        print("✅ Global GEBCO download complete!")
        
    except Exception as e:
        print(f"❌ Global GEBCO download failed: {e}")
        print("   You may need to download GEBCO data manually from:")
        print("   https://www.gebco.net/data_and_products/gridded_bathymetry_data/")

def main():
    """Main download function"""
    
    from config.base import create_global_config
    
    # Create global configuration
    config = create_global_config()
    
    # Create downloader
    downloader = GlobalERA5Downloader(config)
    
    # Download options
    import argparse
    parser = argparse.ArgumentParser(description='Download global ERA5 data')
    parser.add_argument('--start_year', type=int, default=2019, help='Start year')
    parser.add_argument('--end_year', type=int, default=2021, help='End year')
    parser.add_argument('--climatology', action='store_true', help='Download climatology data')
    parser.add_argument('--gebco', action='store_true', help='Download GEBCO bathymetry')
    parser.add_argument('--process', action='store_true', help='Process downloaded data')
    
    args = parser.parse_args()
    
    if args.gebco:
        download_global_gebco_data(config)
    
    if args.climatology:
        downloader.download_climatology_data()
    
    if args.start_year and args.end_year:
        downloader.download_multi_year(args.start_year, args.end_year)
    
    if args.process:
        for year in range(args.start_year, args.end_year + 1):
            downloader.process_year_data(year)
    
    print("\n🎉 Global data download complete!")
    print("   Next steps:")
    print("   1. Process downloaded data with --process flag")
    print("   2. Compute climatology for anomaly calculation")
    print("   3. Train global climate-aware wave model")

if __name__ == "__main__":
    main()