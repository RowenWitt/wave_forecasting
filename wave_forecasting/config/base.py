# config/base.py
"""Base configuration management"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import os

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
                'mean_wave_period'
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
