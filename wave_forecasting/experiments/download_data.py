# experiments/download_data.py
"""Data download utilities"""
import cdsapi
import os
from typing import List
import time

from config.base import DataConfig

class ERA5Downloader:
    """Download ERA5 data systematically"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.client = cdsapi.Client()
        os.makedirs(config.era5_root, exist_ok=True)
    
    def download_year_data(self, year: int, months: List[int] = None):
        """Download a full year of ERA5 data"""
        
        if months is None:
            months = list(range(1, 13))
        
        for month in months:
            print(f"📥 Downloading {year}-{month:02d}...")
            
            try:
                self._download_month_atmospheric(year, month)
                self._download_month_waves(year, month)
                print(f"✅ {year}-{month:02d} complete")
                
            except Exception as e:
                print(f"❌ Failed {year}-{month:02d}: {e}")
                if "queue" in str(e).lower():
                    print("⏳ Waiting for queue...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
    
    def _download_month_atmospheric(self, year: int, month: int):
        """Download atmospheric data for one month"""
        
        output_file = f"{self.config.era5_root}/era5_atmo_{year}{month:02d}.nc"
        
        if os.path.exists(output_file):
            print(f"   Atmospheric data already exists: {output_file}")
            return
        
        self.client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': self.config.atmospheric_vars,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, self.config.time_step_hours)],
            'area': [self.config.lat_bounds[1], self.config.lon_bounds[0], 
                    self.config.lat_bounds[0], self.config.lon_bounds[1]],
            'format': 'netcdf',
        }, output_file)
    
    def _download_month_waves(self, year: int, month: int):
        """Download wave data for one month"""
        
        output_file = f"{self.config.era5_root}/era5_waves_{year}{month:02d}.nc"
        
        if os.path.exists(output_file):
            print(f"   Wave data already exists: {output_file}")
            return
        
        self.client.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': self.config.wave_vars,
            'year': str(year),
            'month': f'{month:02d}',
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, self.config.time_step_hours)],
            'area': [self.config.lat_bounds[1], self.config.lon_bounds[0], 
                    self.config.lat_bounds[0], self.config.lon_bounds[1]],
            'format': 'netcdf',
        }, output_file)
    
    def download_multi_year(self, start_year: int, end_year: int):
        """Download multiple years of data"""
        
        for year in range(start_year, end_year + 1):
            print(f"🗓️  Downloading year {year}...")
            self.download_year_data(year)
            print(f"✅ Year {year} complete!")

def download_gebco_data(config: DataConfig):
    """Download GEBCO bathymetry data"""
    
    import requests
    import xarray as xr
    
    os.makedirs(config.gebco_root, exist_ok=True)
    output_file = f"{config.gebco_root}/gebco_north_pacific.nc"
    
    if os.path.exists(output_file):
        print("GEBCO data already exists")
        return
    
    print("📥 Downloading GEBCO bathymetry...")
    
    try:
        # GEBCO OPeNDAP server
        gebco_url = "https://www.bodc.ac.uk/thredds/dodsC/GEBCO_2023/GEBCO_2023.nc"
        
        gebco_global = xr.open_dataset(gebco_url)
        
        # Determine coordinate names
        lat_coord = 'lat' if 'lat' in gebco_global.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in gebco_global.coords else 'longitude'
        
        # Subset to region
        gebco_subset = gebco_global.sel({
            lat_coord: slice(config.lat_bounds[0], config.lat_bounds[1]),
            lon_coord: slice(config.lon_bounds[0], config.lon_bounds[1])
        })
        
        print("Downloading GEBCO subset...")
        gebco_subset.load()
        gebco_subset.to_netcdf(output_file)
        
        print("✅ GEBCO download complete!")
        
    except Exception as e:
        print(f"❌ GEBCO download failed: {e}")
