# data/preprocessing.py  
"""Data preprocessing and interpolation"""
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import xarray as xr
from typing import List, Optional, Tuple, Dict
import torch


from config.base import DataConfig

class MultiResolutionInterpolator:
    """Handles interpolation from multiple source grids to mesh nodes"""
    
    def __init__(self, era5_atmo: xr.Dataset, era5_waves: xr.Dataset, 
                 gebco: xr.Dataset, config: DataConfig):
        self.era5_atmo = era5_atmo
        self.era5_waves = era5_waves
        self.gebco = gebco
        self.config = config
        
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Pre-build interpolation functions for efficiency"""
        print("Setting up multi-resolution interpolators...")
        
        # ERA5 atmospheric interpolators (0.25°)
        self.atmo_interpolators = {}
        atmo_var_map = {
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            'mean_sea_level_pressure': 'msl'
        }
        
        for var_name in self.config.atmospheric_vars:
            var_key = atmo_var_map.get(var_name, var_name)
            if var_key in self.era5_atmo:
                self.atmo_interpolators[var_key] = RegularGridInterpolator(
                    (self.era5_atmo.latitude.values, self.era5_atmo.longitude.values),
                    self.era5_atmo[var_key].isel(valid_time=0).values,
                    method='linear', bounds_error=False, fill_value=np.nan
                )
        
        # ERA5 wave interpolators (0.5°)
        self.wave_interpolators = {}
        wave_var_map = {
            'significant_height_of_combined_wind_waves_and_swell': 'swh',
            'mean_wave_direction': 'mwd', 
            'mean_wave_period': 'mwp'
        }
        
        for var_name in self.config.wave_vars:
            var_key = wave_var_map.get(var_name, var_name)
            if var_key in self.era5_waves:
                self.wave_interpolators[var_key] = RegularGridInterpolator(
                    (self.era5_waves.latitude.values, self.era5_waves.longitude.values),
                    self.era5_waves[var_key].isel(valid_time=0).values,
                    method='linear', bounds_error=False, fill_value=np.nan
                )
        
        # GEBCO bathymetry interpolators (static)
        self.bathy_interpolators = {}
        for var in ['ocean_depth', 'land_sea_mask', 'depth_gradient', 
                   'shallow_water_mask', 'deep_water_mask']:
            if var in self.gebco:
                self.bathy_interpolators[var] = RegularGridInterpolator(
                    (self.gebco.latitude.values, self.gebco.longitude.values),
                    self.gebco[var].values,
                    method='linear', bounds_error=False, fill_value=0.0
                )
        
        print("✅ Interpolators ready!")
    
    def interpolate_to_points(self, lats: np.ndarray, lons: np.ndarray, 
                            time_idx: int = 0) -> Dict[str, np.ndarray]:
        """Interpolate all variables to given lat/lon points"""
        
        # Update time-dependent interpolators
        self._update_time_interpolators(time_idx)
        
        # Create coordinate pairs
        points = np.column_stack([lats.ravel(), lons.ravel()])
        
        # Interpolate from each dataset
        results = {}
        
        # Atmospheric variables
        for var, interpolator in self.atmo_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        # Wave variables
        for var, interpolator in self.wave_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        # Bathymetry variables (static)
        for var, interpolator in self.bathy_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        return results
    
    def _update_time_interpolators(self, time_idx: int):
        """Update interpolators with new time slice"""
        for var in self.atmo_interpolators:
            if var in self.era5_atmo:
                self.atmo_interpolators[var].values = self.era5_atmo[var].isel(valid_time=time_idx).values
        
        for var in self.wave_interpolators:
            if var in self.era5_waves:
                self.wave_interpolators[var].values = self.era5_waves[var].isel(valid_time=time_idx).values

def clean_features_for_training(features: torch.Tensor) -> torch.Tensor:
    """Clean NaN values and normalize features"""
    
    # Replace NaN and infinite values
    features_clean = torch.where(torch.isnan(features), torch.zeros_like(features), features)
    features_clean = torch.where(torch.isinf(features_clean), torch.zeros_like(features_clean), features_clean)
    
    # Clamp extreme values
    features_clean = torch.clamp(features_clean, -10.0, 10.0)
    
    # Standardization per feature
    for i in range(features_clean.shape[1]):
        col = features_clean[:, i]
        if col.std() > 1e-6:
            features_clean[:, i] = (col - col.mean()) / col.std()
    
    return features_clean
