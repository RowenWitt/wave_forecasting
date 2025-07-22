# config/base.py
"""Base configuration management"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

@dataclass
class GlobalDataConfig:
    """Global data configuration for climate-aware wave forecasting"""
    
    # Data paths
    era5_root: str = "data/era5_global"
    gebco_root: str = "data/gebco_global"
    processed_root: str = "data/processed_global"
    climatology_root: str = "data/climatology"  # For anomaly computation
    
    # Geographic bounds (Global coverage)
    lat_bounds: tuple = (-90.0, 90.0)    # Full global latitude range
    lon_bounds: tuple = (0.0, 360.0)     # Full global longitude range
    
    # Time parameters
    time_step_hours: int = 6
    climatology_years: tuple = (1991, 2020)  # 30-year climatology for anomalies
    
    # Resolution
    spatial_resolution: float = 0.25  # 0.25 degree resolution
    
    # ERA5 variables
    atmospheric_vars: List[str] = None
    wave_vars: List[str] = None
    pressure_level_vars: List[str] = None
    pressure_levels: List[int] = None
    
    def __post_init__(self):
        if self.atmospheric_vars is None:
            self.atmospheric_vars = [
                # Surface winds (original)
                '10m_u_component_of_wind',
                '10m_v_component_of_wind',
                
                # Pressure and temperature fields
                'mean_sea_level_pressure',
                'sea_surface_temperature',
                '2m_temperature',
                
                # Precipitation (for MJO, monsoon patterns)
                'total_precipitation',
                
                # Radiation (for MJO proxy)
                'toa_incident_solar_radiation',  # Top of atmosphere
            ]
        
        if self.wave_vars is None:
            self.wave_vars = [
                # Combined wave parameters (original)
                'significant_height_of_combined_wind_waves_and_swell',
                'mean_wave_direction',
                'mean_wave_period',
                
                # Wind wave components (original)
                'significant_height_of_wind_waves',
                'mean_direction_of_wind_waves', 
                'mean_period_of_wind_waves',
                
                # Swell components (for remote forcing)
                'significant_height_of_total_swell',
                'mean_direction_of_total_swell',
                'mean_period_of_total_swell',
            ]
        
        if self.pressure_level_vars is None:
            self.pressure_level_vars = [
                'u_component_of_wind',     # For jet streams, trade winds
                'v_component_of_wind',     # For meridional flow
                'geopotential',            # For storm tracks, blocking
            ]
        
        if self.pressure_levels is None:
            self.pressure_levels = [
                850,  # Low-level winds (trade winds, monsoons)
                500,  # Mid-level (storm tracks, blocking)
                300,  # Upper-level (jet streams)
            ]
    
    def get_single_level_download_vars(self) -> List[str]:
        """Get variables for single-level download"""
        return self.atmospheric_vars + self.wave_vars
    
    def get_pressure_level_download_vars(self) -> List[str]:
        """Get variables for pressure-level download"""
        return self.pressure_level_vars
    
    def get_feature_mapping(self) -> Dict[str, str]:
        """Map ERA5 variable names to model feature names"""
        mapping = {
            # Surface features
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            'mean_sea_level_pressure': 'slp',
            'sea_surface_temperature': 'sst',
            'total_precipitation': 'precipitation',
            'significant_height_of_combined_wind_waves_and_swell': 'swh',
            'mean_wave_direction': 'mwd',
            'mean_wave_period': 'mwp',
        }
        
        # Add pressure level features
        for level in self.pressure_levels:
            mapping[f'u_component_of_wind_{level}'] = f'u{level}'
            mapping[f'v_component_of_wind_{level}'] = f'v{level}'
            mapping[f'geopotential_{level}'] = f'z{level}'
        
        # Add climate anomaly features (computed later)
        mapping.update({
            'sst_anomaly': 'sst_anomaly',
            'slp_anomaly': 'slp_anomaly', 
            'u850_anomaly': 'u850_anomaly',
            'v850_anomaly': 'v850_anomaly',
            'precip_anomaly': 'precip_anomaly',
            'z500_anomaly': 'z500_anomaly',
        })
        
        return mapping
    
    def get_climate_vars_for_anomalies(self) -> List[str]:
        """Get variables needed for climate anomaly computation"""
        return [
            'sst',           # For SST anomalies (ENSO, IOD, AMO)
            'slp',           # For pressure anomalies (NAO, AO, SAM)
            'u850',          # For wind anomalies (trade winds, monsoons)
            'v850',          # For meridional flow anomalies
            'precipitation', # For precipitation anomalies (MJO, monsoons)
            'z500',          # For geopotential anomalies (blocking, storm tracks)
        ]
    
    def get_target_variables(self) -> List[str]:
        """Get the variables we're predicting"""
        return [
            'significant_height_of_combined_wind_waves_and_swell',
            'mean_wave_direction', 
            'mean_wave_period'
        ]
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        # Check geographic bounds
        if not (-90 <= self.lat_bounds[0] < self.lat_bounds[1] <= 90):
            raise ValueError("Invalid latitude bounds")
        
        if not (0 <= self.lon_bounds[0] < self.lon_bounds[1] <= 360):
            raise ValueError("Invalid longitude bounds")
        
        # Check climatology years
        if self.climatology_years[1] - self.climatology_years[0] < 20:
            raise ValueError("Climatology period should be at least 20 years")
        
        return True

# Helper function for global config
def create_global_config() -> GlobalDataConfig:
    """Create global configuration for climate-aware wave forecasting"""
    
    config = GlobalDataConfig()
    config.validate_config()
    
    print(f"🌍 Global Wave Forecasting Configuration:")
    print(f"   Spatial Domain: {config.lat_bounds}°N, {config.lon_bounds}°E")
    print(f"   Resolution: {config.spatial_resolution}°")
    print(f"   Surface Variables: {len(config.atmospheric_vars + config.wave_vars)}")
    print(f"   Pressure Level Variables: {len(config.pressure_level_vars)}")
    print(f"   Pressure Levels: {config.pressure_levels}")
    print(f"   Climatology Period: {config.climatology_years}")
    
    return config

@dataclass
class DataConfig:
    """Data paths and parameters"""
    era5_root: str = "data/era5"
    gebco_root: str = "data/gebco" 
    processed_root: str = "data/processed"
    
    # Geographic bounds (North Pacific)
    lat_bounds: tuple = (10.0, 60.0)
    lon_bounds: tuple = (120.0, 240.0)
    
    # Time parameters
    time_step_hours: int = 6
    
    # ERA5 variables
    atmospheric_vars: List[str] = None
    wave_vars: List[str] = None
    
    def __post_init__(self):
        if self.atmospheric_vars is None:
            self.atmospheric_vars = [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind', 
                'mean_sea_level_pressure'
            ]
        if self.wave_vars is None:
            self.wave_vars = [
                'significant_height_of_combined_wind_waves_and_swell',
                'mean_wave_direction',
                'mean_wave_period',
                'significant_height_of_wind_waves',       
                'mean_direction_of_wind_waves',           
                'mean_period_of_wind_waves' 
            ]

@dataclass 
class MeshConfig:
    """Mesh parameters"""
    refinement_level: int = 4
    max_edge_distance_km: float = 300.0
    
    @property
    def approx_node_count(self) -> int:
        """Approximate number of nodes for this refinement level"""
        return 10 * (4 ** self.refinement_level)

@dataclass
class ModelConfig:
    """Model architecture parameters"""
    hidden_dim: int = 128
    num_spatial_layers: int = 8
    num_temporal_layers: int = 4
    edge_features: int = 3
    output_features: int = 3  # [swh, mwd, mwp]
    
@dataclass
class TrainingConfig:
    """Training parameters"""
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    mse_weight: float = 1.0
    physics_weight: float = 0.1
    
    # Scheduling
    lr_decay_epochs: int = 10
    lr_decay_factor: float = 0.5

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    data: DataConfig
    mesh: MeshConfig  
    model: ModelConfig
    training: TrainingConfig
    
    # Experiment parameters
    sequence_length: int = 4  # Input timesteps
    forecast_horizon: int = 24  # Hours ahead
    max_training_samples: Optional[int] = None
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
