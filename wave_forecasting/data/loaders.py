# data/loaders.py
"""Data loading and management"""
import xarray as xr
import numpy as np
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os

from config.base import DataConfig

class ERA5DataManager:
    """Manages ERA5 data files across multiple years/months"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.available_files = self._discover_files()
        
    def _discover_files(self) -> Dict[str, List[str]]:
        """Find all available ERA5 files"""
        files = {'atmospheric': [], 'wave': []}
        
        era5_path = Path(self.config.era5_root)
        
        # Find atmospheric files
        atmo_pattern = str(era5_path / "era5_atmo_*.nc")
        files['atmospheric'] = sorted(glob.glob(atmo_pattern))
        
        # Find wave files  
        wave_pattern = str(era5_path / "era5_waves_*.nc")
        files['wave'] = sorted(glob.glob(wave_pattern))
        
        print(f"📁 Found {len(files['atmospheric'])} atmospheric files")
        print(f"📁 Found {len(files['wave'])} wave files")
        
        return files
    
    def load_month_data(self, year: int, month: int) -> Tuple[xr.Dataset, xr.Dataset]:
        """Load data for a specific year/month"""
        atmo_file = f"{self.config.era5_root}/era5_atmo_{year}{month:02d}.nc"
        wave_file = f"{self.config.era5_root}/era5_waves_{year}{month:02d}.nc"
        
        if not os.path.exists(atmo_file) or not os.path.exists(wave_file):
            raise FileNotFoundError(f"Missing data for {year}-{month:02d}")
        
        atmo_data = xr.open_dataset(atmo_file)
        wave_data = xr.open_dataset(wave_file)
        
        return atmo_data, wave_data
    
    def get_time_range(self) -> Tuple[str, str]:
        """Get the available time range"""
        if not self.available_files['atmospheric']:
            return None, None
            
        # Extract dates from first and last files
        first_file = Path(self.available_files['atmospheric'][0]).stem
        last_file = Path(self.available_files['atmospheric'][-1]).stem
        
        # Assuming format: era5_atmo_YYYYMM.nc
        start_date = first_file.split('_')[-1]
        end_date = last_file.split('_')[-1]
        
        return start_date, end_date

class GEBCODataManager:
    """Manages GEBCO bathymetry data"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.processed_file = f"{config.processed_root}/processed_gebco_features.nc"
        
    def load_bathymetry(self) -> xr.Dataset:
        """Load and process GEBCO data"""
        if os.path.exists(self.processed_file):
            print("Loading processed GEBCO data...")
            return xr.open_dataset(self.processed_file)
        
        # Load raw GEBCO and process
        raw_files = glob.glob(f"{self.config.gebco_root}/*.nc")
        if not raw_files:
            raise FileNotFoundError("No GEBCO files found")
        
        print("Processing raw GEBCO data...")
        gebco_raw = xr.open_dataset(raw_files[0])
        processed = self._process_gebco(gebco_raw)
        
        # Save processed version
        os.makedirs(self.config.processed_root, exist_ok=True)
        processed.to_netcdf(self.processed_file)
        
        return processed
    
    def _process_gebco(self, gebco_raw: xr.Dataset) -> xr.Dataset:
        """Convert GEBCO to wave-modeling features"""
        # Standardize coordinate names
        lat_coord = 'lat' if 'lat' in gebco_raw.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in gebco_raw.coords else 'longitude'
        depth_var = 'elevation' if 'elevation' in gebco_raw else 'z'
        
        # Rename for consistency
        gebco_std = gebco_raw.rename({
            depth_var: 'z',
            lat_coord: 'latitude', 
            lon_coord: 'longitude'
        })
        
        # Create wave-relevant features
        ocean_depth = -gebco_std.z.where(gebco_std.z < 0)
        land_sea_mask = (gebco_std.z < 0).astype(float)
        
        # Compute gradients
        depth_grad_lat = ocean_depth.differentiate('latitude')
        depth_grad_lon = ocean_depth.differentiate('longitude')
        depth_gradient = np.sqrt(depth_grad_lat**2 + depth_grad_lon**2)
        
        # Depth categories
        shallow_water = (ocean_depth < 50).astype(float)
        shelf_water = ((ocean_depth >= 50) & (ocean_depth < 500)).astype(float)
        deep_water = (ocean_depth >= 500).astype(float)
        steep_bottom = (depth_gradient > depth_gradient.quantile(0.95)).astype(float)
        
        return xr.Dataset({
            'ocean_depth': ocean_depth,
            'land_sea_mask': land_sea_mask,
            'depth_gradient': depth_gradient,
            'shallow_water_mask': shallow_water,
            'shelf_water_mask': shelf_water,
            'deep_water_mask': deep_water,
            'steep_bottom_mask': steep_bottom,
        })
