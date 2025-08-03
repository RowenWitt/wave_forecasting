#!/usr/bin/env python3
"""
Global Wave Prediction Model V5 - Graph-Aware Cuboid Attention
Complete implementation with all V3 components integrated
Includes bathymetry support, hybrid sampling, and memory-efficient attention
Optimized for 128GB unified memory on Apple M4
"""

import os
import sys
import time
import math
import json
import pickle
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import glob

# Set memory limit for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
import torch.mps
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from bathymetry_model_dual_purpose import DualPurposeBathymetry

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class HybridSamplingConfig:
    """Configuration for V5 wave prediction model with cuboid attention"""
    
    # Data paths
    data_pattern: str = "data/v1_global/processed/v1_era5_2021*.nc"
    cache_dir: str = "cache/global_mesh_v5"
    output_dir: str = "experiments/global_wave_v5_cuboid_attention"
    
    # Geographic coverage (global)
    lat_bounds: tuple = (-90.0, 90.0)
    lon_bounds: tuple = (0.0, 360.0)
    
    # Mesh parameters
    mesh_refinement_level: int = 6  # ~40k nodes globally
    max_edge_distance_km: float = 500.0
    
    # Multiscale edge parameters
    use_multiscale_edges: bool = True
    medium_edge_distance_km: float = 1000.0
    long_edge_distance_km: float = 2000.0
    
    # Cuboid attention parameters (NEW for V5)
    use_cuboid_attention: bool = True
    num_spatial_patches: int = 100  # Number of patches for fine attention
    num_coarse_patches: int = 10   # Number of patches for coarse attention
    patch_overlap_hops: int = 1     # K-hop overlap between patches
    
    # Sampling parameters
    samples_per_epoch: int = 200
    min_samples_per_month: int = 100
    hard_region_boost_factor: float = 2.0
    seasonal_balance: bool = True
    
    # Input features
    input_features: List[str] = field(default_factory=lambda: [
        'tp', 'u10', 'v10', 'msl', 'sst',
        'swh', 'mwd', 'mwp', 'shww',
        'u_850', 'u_500', 'v_850', 'v_500', 'z_850', 'z_500',
        'ocean_depth'
    ])
    num_input_features: int = 16
    
    # Target variables
    target_features: List[str] = field(default_factory=lambda: ['swh', 'mwd', 'mwp'])
    num_output_features: int = 4  # SWH + MWD(cos,sin) + MWP
    
    # Temporal parameters
    sequence_length: int = 4
    prediction_horizon: int = 4
    
    # Model architecture
    hidden_dim: int = 128
    temporal_hidden_dim: int = 64
    num_spatial_layers: int = 4
    num_temporal_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.15
    
    # Training parameters
    batch_size: int = 8  # Increased from 2 since attention is more efficient
    accumulation_steps: int = 1  # Removed accumulation
    num_epochs: int = 100
    base_learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 5.0
    
    # Variable-specific learning rates
    swh_lr_multiplier: float = 0.7
    mwd_lr_multiplier: float = 1.0
    mwp_lr_multiplier: float = 1.3
    
    # Early stopping
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    use_cpu_fallback: bool = True

# ==============================================================================
# HYBRID SAMPLING DATASET WITH BATHYMETRY
# ==============================================================================

class HybridSamplingDataset(Dataset):
    """
    Multi-month dataset with smart sampling strategies:
    - Samples diverse data each epoch
    - Ensures seasonal and regional coverage
    - Focuses on difficult regions
    - Maintains reasonable epoch times
    """
    
    def __init__(self, data_paths: List[str], mesh, config: HybridSamplingConfig):
        self.data_paths = sorted(data_paths)
        self.mesh = mesh
        self.config = config
        
        # Get mesh coordinates for regional sampling
        self.mesh_lats, self.mesh_lons = mesh.vertices_to_lat_lon()
        
        # Create ocean mask for the mesh
        self.ocean_mask = self._create_ocean_mask()
        
        # Initialize sampling metadata
        self.month_metadata = []
        self.total_available_sequences = 0  
        
        print(f"📊 Initializing hybrid sampling dataset...")
        print(f"   Found {len(self.data_paths)} data files")
        
        # Analyze each month
        for path in self.data_paths:
            month_info = self._analyze_month(path)
            self.month_metadata.append(month_info)
            self.total_available_sequences += month_info['n_sequences']
        
        print(f"   Total available sequences: {self.total_available_sequences}")
        print(f"   Samples per epoch: {config.samples_per_epoch}")
        
        # Define regions for targeted sampling
        self.regions = {
            'southern_ocean': {'lat': (-90, -45), 'weight': 2.0},
            'north_pacific': {'lat': (30, 60), 'lon': (120, 240), 'weight': 1.5},
            'mediterranean': {'lat': (30, 45), 'lon': (-5, 40), 'weight': 1.5},
            'tropics': {'lat': (-20, 20), 'weight': 1.0},
            'mid_latitudes': {'lat': (20, 60), 'weight': 1.0},
            'arctic': {'lat': (60, 90), 'weight': 0.5}  # Lower weight (easier)
        }
        
        # Initialize current epoch samples
        self._resample_epoch()
        
        # Cache for currently loaded data
        self.current_path = None
        self.current_data = None

    def _create_ocean_mask(self):
        """Create ocean mask for mesh nodes based on first file's ocean data"""
        print("   Creating ocean mask for mesh nodes...")
        
        # Load first file to get ocean mask
        with xr.open_dataset(self.data_paths[0]) as ds:
            # Use SWH to determine ocean points (NaN = land)
            if 'swh' in ds.variables:
                ocean_data = ds['swh'].isel(valid_time=0).values
            else:
                raise ValueError("No wave variable found to determine ocean mask")
            
            # Create a mask where ocean = 1, land = 0
            ocean_mask_grid = (~np.isnan(ocean_data)).astype(float)
            
            # Interpolate to mesh - but preserve the mask nature
            from scipy.interpolate import RegularGridInterpolator
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            interpolator = RegularGridInterpolator(
                (lats, lons), ocean_mask_grid,
                method='nearest',  # Use nearest to preserve mask
                bounds_error=False, 
                fill_value=0.0
            )
            
            points = np.column_stack([self.mesh_lats, self.mesh_lons])
            ocean_mask = interpolator(points) > 0.5  # Boolean mask
            
            n_ocean = ocean_mask.sum()
            print(f"   Ocean nodes: {n_ocean}/{len(ocean_mask)} ({n_ocean/len(ocean_mask)*100:.1f}%)")
            
            return ocean_mask
    
    def _analyze_month(self, path: str) -> Dict:
        """Analyze a month's data file"""
        with xr.open_dataset(path) as ds:
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            n_timesteps = len(ds[time_dim])
            n_sequences = n_timesteps - self.config.sequence_length - self.config.prediction_horizon
            
            # Extract month/season info
            month_num = int(Path(path).stem.split('_')[-1][-2:])  # Last 2 digits
            season = self._get_season(month_num)
            
            return {
                'path': path,
                'n_sequences': n_sequences,
                'month': month_num,
                'season': season,
                'filename': Path(path).name
            }
    
    def _get_season(self, month: int) -> str:
        """Get season from month number"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_region_weight(self, lat: float, lon: float) -> float:
        """Get sampling weight based on region difficulty"""
        for region_name, region_info in self.regions.items():
            lat_match = region_info['lat'][0] <= lat <= region_info['lat'][1]
            
            if 'lon' in region_info:
                lon_match = region_info['lon'][0] <= lon <= region_info['lon'][1]
            else:
                lon_match = True
            
            if lat_match and lon_match:
                return region_info['weight']
        
        return 1.0  # Default weight
    
    def _resample_epoch(self):
        """Create a new sampling for this epoch"""
        print(f"\n🎲 Resampling epoch data...")
        
        # Calculate samples per month with minimum guarantee
        base_samples_per_month = self.config.samples_per_epoch // len(self.month_metadata)
        min_samples = self.config.min_samples_per_month
        samples_per_month = max(base_samples_per_month, min_samples)
        
        # Ensure we don't exceed target
        if samples_per_month * len(self.month_metadata) > self.config.samples_per_epoch:
            samples_per_month = self.config.samples_per_epoch // len(self.month_metadata)
        
        self.epoch_samples = []
        
        # Sample from each month
        for month_info in self.month_metadata:
            n_samples = min(samples_per_month, month_info['n_sequences'])
            
            # Random indices from this month
            indices = np.random.choice(month_info['n_sequences'], size=n_samples, replace=False)
            
            for idx in indices:
                self.epoch_samples.append({
                    'path': month_info['path'],
                    'local_idx': idx,
                    'month': month_info['month'],
                    'season': month_info['season']
                })
        
        # If we have room, add more samples focusing on difficult regions
        remaining_budget = self.config.samples_per_epoch - len(self.epoch_samples)
        if remaining_budget > 0:
            # Add weighted samples
            additional_samples = self._sample_difficult_regions(remaining_budget)
            self.epoch_samples.extend(additional_samples)
        
        # Shuffle for random ordering
        np.random.shuffle(self.epoch_samples)
        
        print(f"   Sampled {len(self.epoch_samples)} sequences")
        
        # Print seasonal distribution
        season_counts = {}
        for sample in self.epoch_samples:
            season = sample['season']
            season_counts[season] = season_counts.get(season, 0) + 1
        print(f"   Seasonal distribution: {season_counts}")
    
    def _sample_difficult_regions(self, n_samples: int) -> List[Dict]:
        """Sample more from difficult regions"""
        weighted_samples = []
        
        # Build weighted sampling pool
        sampling_pool = []
        weights = []
        
        for month_info in self.month_metadata:
            # For each sequence in this month, assign a weight
            # In practice, we'll sample a subset and estimate weights
            n_test = min(100, month_info['n_sequences'])
            test_indices = np.random.choice(month_info['n_sequences'], n_test, replace=False)
            
            for idx in test_indices:
                # Estimate region from sequence (simplified)
                # In real implementation, would load and check actual data
                sampling_pool.append({
                    'path': month_info['path'],
                    'local_idx': idx,
                    'month': month_info['month'],
                    'season': month_info['season']
                })
                
                # Assign weight based on approximate latitude
                # (This is simplified - in practice would check actual data)
                estimated_lat = np.random.uniform(-90, 90)
                weight = self._get_region_weight(estimated_lat, 0)
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample based on weights
        if len(sampling_pool) > 0:
            sample_indices = np.random.choice(
                len(sampling_pool), 
                size=min(n_samples, len(sampling_pool)), 
                replace=True,
                p=weights
            )
            
            for idx in sample_indices:
                weighted_samples.append(sampling_pool[idx])
        
        return weighted_samples
    
    def _load_data(self, path: str):
        """Load data file with caching"""
        if self.current_path != path:
            if self.current_data is not None:
                self.current_data.close()
            
            self.current_data = xr.open_dataset(path)
            self.current_path = path
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        sample_info = self.epoch_samples[idx]
        
        # Load data file if needed
        self._load_data(sample_info['path'])
        
        # Extract sequence
        local_idx = sample_info['local_idx']
        
        # Get time dimension
        time_dim = 'time' if 'time' in self.current_data.dims else 'valid_time'
        
        # Extract input features
        input_features = []
        
        for t in range(self.config.sequence_length):
            t_idx = local_idx + t
            timestep_features = []
            
            for feat in self.config.input_features:
                if feat == 'ocean_depth':
                    # Skip if not available
                    timestep_features.append(np.zeros(len(self.mesh.vertices), dtype=np.float32))
                    continue
                    
                if feat in self.current_data.variables:
                    if time_dim in self.current_data[feat].dims:
                        field_data = self.current_data[feat].isel(**{time_dim: t_idx}).values
                    else:
                        field_data = self.current_data[feat].values
                    
                    # Determine if this is an ocean variable
                    is_ocean_var = feat in ['swh', 'mwd', 'mwp', 'shww', 'sst']
                    
                    # Interpolate to mesh with ocean awareness
                    mesh_data = self._interpolate_to_mesh(field_data, is_ocean_variable=is_ocean_var)
                    timestep_features.append(mesh_data.astype(np.float32))
                else:
                    timestep_features.append(np.zeros(len(self.mesh.vertices), dtype=np.float32))
            
            input_features.append(np.stack(timestep_features, axis=-1))
        
        inputs = np.stack(input_features, axis=0)
        
        # Extract targets
        target_idx = local_idx + self.config.sequence_length
        target_features = []
        
        for feat in self.config.target_features:
            if feat in self.current_data.variables:
                if time_dim in self.current_data[feat].dims:
                    field_data = self.current_data[feat].isel(**{time_dim: target_idx}).values
                else:
                    field_data = self.current_data[feat].values
                
                mesh_data = self._interpolate_to_mesh(field_data, is_ocean_variable=True)
                target_features.append(mesh_data.astype(np.float32))
        
        targets = np.stack(target_features, axis=-1)
        
        return {
            'input': torch.FloatTensor(inputs),
            'target': torch.FloatTensor(targets),
            'single_step_target': torch.FloatTensor(targets),
            'metadata': sample_info  # Include metadata for analysis
        }
    
    def _interpolate_to_mesh(self, field_data: np.ndarray, is_ocean_variable: bool = False) -> np.ndarray:
        """Interpolate regular grid data to icosahedral mesh points"""
        from scipy.interpolate import RegularGridInterpolator
        
        lats = self.current_data.latitude.values
        lons = self.current_data.longitude.values
        
        if is_ocean_variable:
            # For ocean variables, use nearest-neighbor at boundaries
            # to avoid spreading NaN into ocean areas
            
            # First, create a filled version for stable interpolation
            field_data_filled = field_data.copy()
            ocean_mask = ~np.isnan(field_data)
            
            if ocean_mask.any():
                # Use nearest-neighbor filling from ocean values
                from scipy.ndimage import distance_transform_edt
                indices = distance_transform_edt(~ocean_mask, return_indices=True, return_distances=False)
                field_data_filled = field_data[indices[0], indices[1]]
            else:
                # No ocean data at all - shouldn't happen
                field_data_filled = np.zeros_like(field_data)
            
            interpolator = RegularGridInterpolator(
                (lats, lons), field_data_filled,
                method='linear', bounds_error=False, fill_value=0.0
            )
        else:
            # For atmospheric variables, simple fill is fine
            field_data_filled = np.nan_to_num(field_data, nan=0.0)
            
            interpolator = RegularGridInterpolator(
                (lats, lons), field_data_filled,
                method='linear', bounds_error=False, fill_value=0.0
            )
        
        points = np.column_stack([self.mesh_lats, self.mesh_lons])
        interpolated = interpolator(points)
        
        # Apply ocean mask if this is an ocean variable
        if is_ocean_variable and hasattr(self, 'ocean_mask'):
            interpolated[~self.ocean_mask] = 0.0
        
        return interpolated
    
    def on_epoch_end(self):
        """Resample data for next epoch"""
        self._resample_epoch()

class BathymetryEnabledHybridSamplingDataset(HybridSamplingDataset):
    """
    HybridSamplingDataset with integrated bathymetry support
    Extends the base class to add ocean masking and proper depth features
    """
    
    def __init__(self, data_paths: List[str], mesh, config, gebco_path: str = None):
        """
        Initialize dataset with optional bathymetry support
        
        Args:
            data_paths: List of data file paths
            mesh: Icosahedral mesh object
            config: HybridSamplingConfig
            gebco_path: Path to GEBCO bathymetry file (optional)
        """
        # Initialize parent class first
        super().__init__(data_paths, mesh, config)
        
        # Add bathymetry support if path provided
        if gebco_path:
            print("\n🌊 Initializing bathymetry support...")
            
            # Create dual-purpose bathymetry handler
            self.bathymetry = DualPurposeBathymetry(
                gebco_path=gebco_path,
                cache_dir=f"cache/bathymetry_dual/{Path(config.output_dir).name}",
                resolution=0.1,
                ocean_threshold=-10.0,
                max_depth=5000.0,
                normalize_depth=True
            )
            
            # Get bathymetry data for mesh
            bath_data = self.bathymetry.get_mesh_bathymetry(self.mesh)
            self.ocean_mask = bath_data['mask']
            self.mesh_bathymetry = bath_data['normalized_depth']
            
            print(f"🌊 Bathymetry integrated: mask + feature")
            print(f"   Ocean nodes: {self.ocean_mask.sum():,}/{len(self.ocean_mask):,}")
        else:
            print("⚠️  No GEBCO path provided - using data-based ocean detection")
            self.ocean_mask = None
            self.mesh_bathymetry = None
            self.bathymetry = None
    
    def _interpolate_to_mesh(self, field_data: np.ndarray, 
                            variable_name: Optional[str] = None,
                            is_ocean_variable: bool = False) -> np.ndarray:
        """
        Enhanced interpolation with ocean mask support
        
        Args:
            field_data: Regular grid data to interpolate
            variable_name: Name of the variable (for special handling)
            is_ocean_variable: Whether this is an ocean-only variable
            
        Returns:
            Interpolated data on mesh nodes
        """
        # Special handling for ocean_depth - use pre-computed bathymetry
        if variable_name == 'ocean_depth' and self.mesh_bathymetry is not None:
            return self.mesh_bathymetry
        
        # Regular interpolation using parent method
        interpolated = super()._interpolate_to_mesh(field_data, is_ocean_variable=is_ocean_variable)
        
        # Apply ocean mask for ocean variables
        if self.ocean_mask is not None and (is_ocean_variable or 
            variable_name in ['swh', 'mwd', 'mwp', 'shww', 'sst']):
            # Zero out land nodes
            interpolated[~self.ocean_mask] = 0.0
        
        return interpolated
    
    def __getitem__(self, idx):
        """
        Get a sample with enhanced bathymetry support
        """
        sample_info = self.epoch_samples[idx]
        
        # Load data file if needed
        self._load_data(sample_info['path'])
        
        # Extract sequence
        local_idx = sample_info['local_idx']
        
        # Get time dimension
        time_dim = 'time' if 'time' in self.current_data.dims else 'valid_time'
        
        # Extract input features
        input_features = []
        
        for t in range(self.config.sequence_length):
            t_idx = local_idx + t
            timestep_features = []
            
            for feat in self.config.input_features:
                if feat in self.current_data.variables:
                    if time_dim in self.current_data[feat].dims:
                        field_data = self.current_data[feat].isel(**{time_dim: t_idx}).values
                    else:
                        field_data = self.current_data[feat].values
                    
                    # Determine if this is an ocean variable
                    is_ocean_var = feat in ['swh', 'mwd', 'mwp', 'shww', 'sst']
                    
                    # Interpolate to mesh with ocean awareness
                    mesh_data = self._interpolate_to_mesh(
                        field_data, 
                        variable_name=feat,
                        is_ocean_variable=is_ocean_var
                    )
                    timestep_features.append(mesh_data.astype(np.float32))
                else:
                    # Handle ocean_depth specially
                    if feat == 'ocean_depth' and self.mesh_bathymetry is not None:
                        timestep_features.append(self.mesh_bathymetry.astype(np.float32))
                    else:
                        timestep_features.append(np.zeros(len(self.mesh.vertices), dtype=np.float32))
            
            input_features.append(np.stack(timestep_features, axis=-1))
        
        inputs = np.stack(input_features, axis=0)
        
        # Extract targets
        target_idx = local_idx + self.config.sequence_length
        target_features = []
        
        for feat in self.config.target_features:
            if feat in self.current_data.variables:
                if time_dim in self.current_data[feat].dims:
                    field_data = self.current_data[feat].isel(**{time_dim: target_idx}).values
                else:
                    field_data = self.current_data[feat].values
                
                mesh_data = self._interpolate_to_mesh(field_data, is_ocean_variable=True)
                target_features.append(mesh_data.astype(np.float32))
        
        targets = np.stack(target_features, axis=-1)
        
        return {
            'input': torch.FloatTensor(inputs),
            'target': torch.FloatTensor(targets),
            'single_step_target': torch.FloatTensor(targets),
            'metadata': sample_info
        }
    
    def get_ocean_statistics(self) -> Dict[str, float]:
        """
        Get statistics about ocean coverage
        
        Returns:
            Dictionary with ocean statistics
        """
        if self.ocean_mask is None:
            return {"ocean_percentage": 0.0, "ocean_nodes": 0, "total_nodes": 0}
        
        ocean_nodes = int(self.ocean_mask.sum())
        total_nodes = len(self.ocean_mask)
        ocean_percentage = ocean_nodes / total_nodes * 100
        
        return {
            "ocean_percentage": ocean_percentage,
            "ocean_nodes": ocean_nodes,
            "total_nodes": total_nodes,
            "mean_depth": float(self.mesh_bathymetry[self.ocean_mask].mean()) if ocean_nodes > 0 else 0.0,
            "max_depth": float(self.mesh_bathymetry.max())
        }

# ==============================================================================
# CUSTOM SAMPLER FOR BATCH DIVERSITY
# ==============================================================================

class DiverseBatchSampler(Sampler):
    """Ensures each batch has diverse seasonal/regional representation"""
    
    def __init__(self, dataset: HybridSamplingDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Make sure epoch_samples exist
        if not hasattr(dataset, 'epoch_samples') or len(dataset.epoch_samples) == 0:
            raise ValueError("Dataset must have epoch_samples initialized. Call dataset._resample_epoch() first.")
        
        self._prepare_season_groups()
    
    def _prepare_season_groups(self):
        """Prepare season groups from current epoch samples"""
        # Group samples by season
        self.season_groups = {'winter': [], 'spring': [], 'summer': [], 'fall': []}
        
        for i, sample in enumerate(self.dataset.epoch_samples):
            season = sample.get('season', 'winter')  # Default to winter if missing
            self.season_groups[season].append(i)
        
        # Remove empty seasons
        self.season_groups = {k: v for k, v in self.season_groups.items() if len(v) > 0}
        
        # Debug info
        print(f"   DiverseBatchSampler initialized:")
        for season, indices in self.season_groups.items():
            print(f"      {season}: {len(indices)} samples")
    
    def __iter__(self):
        # Make copies of season groups to modify
        available_samples = {season: list(indices) for season, indices in self.season_groups.items()}
        
        # Shuffle within each season
        for season_indices in available_samples.values():
            np.random.shuffle(season_indices)
        
        # Generate batches
        while sum(len(indices) for indices in available_samples.values()) >= self.batch_size:
            batch = []
            
            # Try to get one sample from each season
            seasons = list(available_samples.keys())
            np.random.shuffle(seasons)  # Randomize season order
            
            for season in seasons:
                if available_samples[season] and len(batch) < self.batch_size:
                    batch.append(available_samples[season].pop())
            
            # Fill remaining slots randomly
            if len(batch) < self.batch_size:
                # Collect all remaining indices
                all_remaining = []
                for indices in available_samples.values():
                    all_remaining.extend(indices)
                
                # Shuffle and take what we need
                np.random.shuffle(all_remaining)
                needed = self.batch_size - len(batch)
                batch.extend(all_remaining[:needed])
                
                # Remove used indices
                for idx in all_remaining[:needed]:
                    for season_indices in available_samples.values():
                        if idx in season_indices:
                            season_indices.remove(idx)
                            break
            
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

# ==============================================================================
# MESH GENERATION
# ==============================================================================

class MultiscaleGlobalIcosahedralMesh:
    """Global icosahedral mesh with multiscale edge connectivity"""
    
    def __init__(self, refinement_level: int, config: HybridSamplingConfig, cache_dir: str = "cache/global_mesh_v5"):
        self.refinement_level = refinement_level
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.vertices = None
        self.faces = None
        self.edges = None
        self.multiscale_edges = None
        self.edge_attributes = None
        
        self._build_or_load_mesh()
    
    def _get_cache_key(self) -> str:
        """Generate cache key for this mesh configuration including multiscale params"""
        config_str = (f"global_multiscale_ico_level_{self.refinement_level}_"
                     f"local_{int(self.config.max_edge_distance_km)}_"
                     f"medium_{int(self.config.medium_edge_distance_km)}_"
                     f"long_{int(self.config.long_edge_distance_km)}")
        return hashlib.md5(config_str.encode()).hexdigest()[:12] + ".pkl"
    
    def _build_or_load_mesh(self):
        """Build mesh or load from cache"""
        cache_file = self.cache_dir / self._get_cache_key()
        
        if cache_file.exists():
            print(f"📂 Loading cached multiscale mesh from: {cache_file}")
            with open(cache_file, 'rb') as f:
                mesh_data = pickle.load(f)
            self.vertices = mesh_data['vertices']
            self.faces = mesh_data['faces']
            self.edges = mesh_data['edges']
            self.multiscale_edges = mesh_data['multiscale_edges']
            self.edge_attributes = mesh_data['edge_attributes']
            print(f"✅ Multiscale mesh loaded: {len(self.vertices)} vertices")
        else:
            print(f"🔨 Building multiscale global icosahedral mesh (level {self.refinement_level})...")
            self._build_mesh()
            
            # Cache the mesh
            mesh_data = {
                'vertices': self.vertices,
                'faces': self.faces,
                'edges': self.edges,
                'multiscale_edges': self.multiscale_edges,
                'edge_attributes': self.edge_attributes,
                'config': {
                    'refinement_level': self.refinement_level,
                    'max_edge_distance_km': self.config.max_edge_distance_km,
                    'medium_edge_distance_km': self.config.medium_edge_distance_km,
                    'long_edge_distance_km': self.config.long_edge_distance_km
                }
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(mesh_data, f)
            print(f"💾 Multiscale mesh cached to: {cache_file}")
    
    def _build_mesh(self):
        """Build the icosahedral mesh with multiscale connectivity"""
        # Start with base icosahedron
        vertices, faces = self._create_base_icosahedron()
        
        # Refine iteratively
        for level in range(self.refinement_level):
            vertices, faces = self._subdivide_mesh(vertices, faces)
            print(f"  Level {level+1}: {len(vertices)} vertices")
        
        # Project to unit sphere
        self.vertices = self._normalize_to_sphere(vertices)
        self.faces = faces
        self.edges = self._create_edges_from_faces(faces)
        
        # Create multiscale edges
        print(f"  Building multiscale edge connectivity...")
        self.multiscale_edges = self._create_multiscale_edges()
        
        # Compute edge attributes for all edge types
        print(f"  Computing edge attributes...")
        self.edge_attributes = self._compute_edge_attributes()
        
        print(f"✅ Multiscale mesh complete:")
        print(f"   Vertices: {len(self.vertices)}")
        print(f"   Local edges: {len(self.multiscale_edges['local'])}")
        print(f"   Medium edges: {len(self.multiscale_edges['medium'])}")
        print(f"   Long edges: {len(self.multiscale_edges['long'])}")
    
    def _create_base_icosahedron(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create the basic 12-vertex icosahedron"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        vertices = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ], dtype=float)
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        
        return vertices, faces
    
    def _subdivide_mesh(self, vertices, faces):
        """Subdivide each triangle into 4 smaller triangles"""
        new_vertices = list(vertices)
        new_faces = []
        edge_midpoints = {}
        
        def get_midpoint(v1_idx, v2_idx):
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge not in edge_midpoints:
                midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2
                edge_midpoints[edge] = len(new_vertices)
                new_vertices.append(midpoint)
            return edge_midpoints[edge]
        
        for face in faces:
            v1, v2, v3 = face
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m31 = get_midpoint(v3, v1)
            
            new_faces.extend([
                [v1, m12, m31], [v2, m23, m12],
                [v3, m31, m23], [m12, m23, m31]
            ])
        
        return np.array(new_vertices), np.array(new_faces)
    
    def _normalize_to_sphere(self, vertices):
        """Project vertices onto unit sphere"""
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms
    
    def _create_edges_from_faces(self, faces):
        """Extract unique edges from faces"""
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                edges.add(edge)
        return np.array(list(edges))
    
    def _create_multiscale_edges(self) -> Dict[str, np.ndarray]:
        """Create multiscale edge connectivity efficiently"""
        n_vertices = len(self.vertices)
        
        print(f"    Building multiscale edges for {n_vertices} vertices...")
        
        # Use the base edges as local edges
        local_edges_set = set()
        for edge in self.edges:
            local_edges_set.add((edge[0], edge[1]))
            local_edges_set.add((edge[1], edge[0]))  # Bidirectional
        
        local_edges = np.array(list(local_edges_set))
        
        # For medium and long-range edges, use a more efficient approach
        # Only compute for a subset of vertices to keep memory manageable
        
        # Convert vertices to lat/lon once
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        
        # Compute pairwise distances more efficiently using dot products
        # For unit sphere: distance = arccos(dot product)
        
        medium_edges = []
        long_edges = []
        
        # Sample vertices for medium/long range connections
        # Use every Nth vertex to reduce computation
        stride = max(1, n_vertices // 5000)  # Limit to ~5000 vertices for long-range
        sampled_indices = np.arange(0, n_vertices, stride)
        
        print(f"    Computing medium/long edges for {len(sampled_indices)} sampled vertices...")
        
        for i, idx1 in enumerate(sampled_indices):
            if i % 100 == 0:
                print(f"      Processing {i}/{len(sampled_indices)}...")
            
            # Compute dot products with all other vertices
            dots = np.dot(self.vertices, self.vertices[idx1])
            dots = np.clip(dots, -1, 1)  # Numerical stability
            
            # Angular distances in radians
            angles = np.arccos(dots)
            
            # Convert to km (Earth radius = 6371 km)
            distances = angles * 6371
            
            # Find vertices in medium range
            medium_mask = (distances > self.config.max_edge_distance_km) & \
                         (distances <= self.config.medium_edge_distance_km)
            medium_indices = np.where(medium_mask)[0]
            
            # Subsample if too many
            if len(medium_indices) > 50:
                medium_indices = np.random.choice(medium_indices, 50, replace=False)
            
            for idx2 in medium_indices:
                medium_edges.append([idx1, idx2])
                medium_edges.append([idx2, idx1])
            
            # Find vertices in long range
            long_mask = (distances > self.config.medium_edge_distance_km) & \
                       (distances <= self.config.long_edge_distance_km)
            long_indices = np.where(long_mask)[0]
            
            # Subsample if too many
            if len(long_indices) > 20:
                long_indices = np.random.choice(long_indices, 20, replace=False)
            
            for idx2 in long_indices:
                long_edges.append([idx1, idx2])
                long_edges.append([idx2, idx1])
        
        # Convert to arrays
        multiscale_edges = {
            'local': local_edges,
            'medium': np.array(medium_edges) if medium_edges else np.empty((0, 2), dtype=int),
            'long': np.array(long_edges) if long_edges else np.empty((0, 2), dtype=int)
        }
        
        return multiscale_edges
    
    def _compute_edge_attributes(self) -> Dict[str, np.ndarray]:
        """Compute attributes for all edge types"""
        lats, lons = self.vertices_to_lat_lon()
        edge_attrs = {}
        
        for edge_type, edges in self.multiscale_edges.items():
            if len(edges) == 0:
                edge_attrs[edge_type] = np.empty((0, 3), dtype=np.float32)
                continue
            
            attrs = []
            for edge in edges:
                i, j = edge
                lat1, lon1 = lats[i], lons[i]
                lat2, lon2 = lats[j], lons[j]
                
                # Compute distance
                lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
                lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
                
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                
                a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371 * c
                
                # Edge features: [distance_km/1000, lat_diff/10, lon_diff/10]
                attrs.append([
                    distance / 1000.0,
                    (lat2 - lat1) / 10.0,
                    (lon2 - lon1) / 10.0
                ])
            
            edge_attrs[edge_type] = np.array(attrs, dtype=np.float32)
        
        return edge_attrs
    
    def vertices_to_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D vertices to lat/lon coordinates"""
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        
        lat = np.arcsin(z) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        lon = np.where(lon < 0, lon + 360, lon)  # [0, 360)
        
        return lat, lon
    
    def get_combined_edge_index_and_attr(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, slice]]:
        """Get combined edge index and attributes for all edge types"""
        # Combine all edges
        all_edges = []
        all_attrs = []
        edge_slices = {}
        
        current_idx = 0
        for edge_type in ['local', 'medium', 'long']:
            edges = self.multiscale_edges[edge_type]
            attrs = self.edge_attributes[edge_type]
            
            if len(edges) > 0:
                all_edges.append(edges)
                all_attrs.append(attrs)
                edge_slices[edge_type] = slice(current_idx, current_idx + len(edges))
                current_idx += len(edges)
        
        # Convert to tensors
        if all_edges:
            edge_index = torch.tensor(np.vstack(all_edges).T, dtype=torch.long)
            edge_attr = torch.tensor(np.vstack(all_attrs), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)
        
        return edge_index, edge_attr, edge_slices

# ==============================================================================
# NORMALIZERS
# ==============================================================================

class CircularNormalizer:
    """Handle circular wave direction normalization"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, angles_deg: np.ndarray):
        """Fit normalizer (no-op for circular)"""
        self.fitted = True
    
    def transform(self, angles_deg: np.ndarray) -> np.ndarray:
        """Transform angles to [cos, sin] representation"""
        angles_rad = np.deg2rad(angles_deg)
        cos_vals = np.cos(angles_rad)
        sin_vals = np.sin(angles_rad)
        return np.column_stack([cos_vals, sin_vals])
    
    def inverse_transform(self, cos_sin: np.ndarray) -> np.ndarray:
        """Transform [cos, sin] back to angles"""
        angles_rad = np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
        angles_deg = np.rad2deg(angles_rad)
        return np.where(angles_deg < 0, angles_deg + 360, angles_deg)

class VariableSpecificNormalizer:
    """Normalizer for all variables with circular MWD handling"""
    
    def __init__(self):
        self.swh_scaler = RobustScaler()
        self.mwd_normalizer = CircularNormalizer()
        self.mwp_scaler = RobustScaler()
        self.fitted = False
    
    def fit(self, targets: np.ndarray):
        """Fit normalizers on target data"""
        swh = targets[:, 0:1]
        mwd = targets[:, 1]
        mwp = targets[:, 2:3]
        
        self.swh_scaler.fit(swh)
        self.mwd_normalizer.fit(mwd)
        self.mwp_scaler.fit(mwp)
        self.fitted = True
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets to normalized form"""
        swh = targets[:, 0:1]
        mwd = targets[:, 1]
        mwp = targets[:, 2:3]
        
        swh_norm = self.swh_scaler.transform(swh)
        mwd_norm = self.mwd_normalizer.transform(mwd)  # Returns [cos, sin]
        mwp_norm = self.mwp_scaler.transform(mwp)
        
        # Concatenate: [SWH, MWD_cos, MWD_sin, MWP]
        return np.concatenate([swh_norm, mwd_norm, mwp_norm], axis=1)
    
    def inverse_transform_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Transform back to original scale"""
        swh_norm = normalized[:, 0:1]
        mwd_norm = normalized[:, 1:3]  # [cos, sin]
        mwp_norm = normalized[:, 3:4]
        
        swh = self.swh_scaler.inverse_transform(swh_norm)
        mwd = self.mwd_normalizer.inverse_transform(mwd_norm)
        mwp = self.mwp_scaler.inverse_transform(mwp_norm)
        
        return np.column_stack([swh.flatten(), mwd, mwp.flatten()])

# ==============================================================================
# GRAPH-AWARE CUBOID ATTENTION
# ==============================================================================

class GraphPartitioner:
    """Partition icosahedral mesh into patches using graph clustering"""
    
    def __init__(self, mesh, num_patches: int, overlap_hops: int = 1):
        self.mesh = mesh
        self.num_patches = num_patches
        self.overlap_hops = overlap_hops
        self._partitions = None
        
    def get_partitions(self):
        """Get or create graph partitions"""
        if self._partitions is not None:
            return self._partitions
            
        print(f"   Creating {self.num_patches} graph partitions...")
        
        # Use mesh vertices for clustering
        vertices = self.mesh.vertices
        
        # K-means clustering on 3D coordinates
        kmeans = KMeans(n_clusters=self.num_patches, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(vertices)
        
        # Build adjacency list from edges
        adjacency = {i: set() for i in range(len(vertices))}
        for edge in self.mesh.edges:
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
        
        # Create patches with overlap
        patches = []
        for patch_id in range(self.num_patches):
            # Core nodes
            core_nodes = np.where(cluster_labels == patch_id)[0].tolist()
            
            # Add k-hop neighbors as overlap
            overlap_nodes = set()
            current_frontier = set(core_nodes)
            
            for hop in range(self.overlap_hops):
                next_frontier = set()
                for node in current_frontier:
                    for neighbor in adjacency[node]:
                        if neighbor not in core_nodes:
                            next_frontier.add(neighbor)
                overlap_nodes.update(next_frontier)
                current_frontier = next_frontier
            
            patches.append({
                'id': patch_id,
                'core': core_nodes,
                'overlap': list(overlap_nodes),
                'all': core_nodes + list(overlap_nodes)
            })
        
        self._partitions = patches
        
        # Print statistics
        core_sizes = [len(p['core']) for p in patches]
        total_sizes = [len(p['all']) for p in patches]
        print(f"   Partition stats: core nodes {np.mean(core_sizes):.0f}±{np.std(core_sizes):.0f}, "
              f"total nodes {np.mean(total_sizes):.0f}±{np.std(total_sizes):.0f}")
        
        return patches

class GraphAwareCuboidAttention(nn.Module):
    """Efficient attention within graph-based patches"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, 
                 partitioner: Optional[GraphPartitioner] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.partitioner = partitioner
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
        """
        Apply attention within graph patches
        
        Args:
            x: Input tensor [batch, nodes, hidden] or [nodes, hidden]
            batch_first: Whether batch dimension is first
            
        Returns:
            Output tensor with same shape as input
        """
        # Handle both batched and unbatched input
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Get partitions
        if self.partitioner is None:
            # Fallback: treat all nodes as one patch (not recommended for large graphs)
            patches = [{'all': list(range(num_nodes)), 'core': list(range(num_nodes))}]
        else:
            patches = self.partitioner.get_partitions()
        
        # Initialize output
        output = torch.zeros_like(x)
        update_count = torch.zeros(num_nodes, device=x.device)
        
        # Process each patch
        for patch in patches:
            patch_indices = patch['all']
            core_indices = patch['core']
            
            if len(patch_indices) == 0:
                continue
                
            # Extract patch features
            patch_x = x[:, patch_indices, :]  # [batch, patch_nodes, hidden]
            
            # Compute Q, K, V
            Q = self.q_proj(patch_x).view(batch_size, len(patch_indices), self.num_heads, self.head_dim)
            K = self.k_proj(patch_x).view(batch_size, len(patch_indices), self.num_heads, self.head_dim)
            V = self.v_proj(patch_x).view(batch_size, len(patch_indices), self.num_heads, self.head_dim)
            
            # Transpose for attention computation
            Q = Q.transpose(1, 2)  # [batch, heads, patch_nodes, head_dim]
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention
            attended = torch.matmul(attn_weights, V)  # [batch, heads, patch_nodes, head_dim]
            attended = attended.transpose(1, 2).contiguous()  # [batch, patch_nodes, heads, head_dim]
            attended = attended.view(batch_size, len(patch_indices), hidden_dim)
            
            # Project output
            patch_output = self.out_proj(attended)
            
            # Update only core nodes (overlap nodes will be averaged)
            core_mask = torch.zeros(len(patch_indices), dtype=torch.bool, device=x.device)
            for i, idx in enumerate(patch_indices):
                if idx in core_indices:
                    core_mask[i] = True
                    output[:, idx, :] += patch_output[:, i, :]
                    update_count[idx] += 1
        
        # Average overlapping updates
        output = output / update_count.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + output)
        
        return output.squeeze(0) if squeeze_output else output

class HierarchicalCuboidAttention(nn.Module):
    """Two-level attention: fine local patches + coarse global patches"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8,
                 fine_partitioner: Optional[GraphPartitioner] = None,
                 coarse_partitioner: Optional[GraphPartitioner] = None):
        super().__init__()
        
        # Fine-grained local attention
        self.fine_attention = GraphAwareCuboidAttention(
            hidden_dim, num_heads, fine_partitioner
        )
        
        # Coarse global attention
        self.coarse_attention = GraphAwareCuboidAttention(
            hidden_dim, num_heads, coarse_partitioner
        )
        
        # Learned gating between fine and coarse
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical attention"""
        # Fine-grained attention
        x_fine = self.fine_attention(x)
        
        # Coarse attention
        x_coarse = self.coarse_attention(x)
        
        # Adaptive combination
        gate_input = torch.cat([x_fine, x_coarse], dim=-1)
        gate_value = self.gate(gate_input)
        
        output = gate_value * x_fine + (1 - gate_value) * x_coarse
        
        return output

# ==============================================================================
# MODEL COMPONENTS
# ==============================================================================

class MultiscaleMessageLayer(nn.Module):
    """Message passing layer that handles multiple edge types"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Separate message functions for each edge type
        self.local_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.medium_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.long_message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregation and update
        self.update_gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                edge_slices: Dict[str, slice]) -> torch.Tensor:
        
        num_nodes = x.size(0)
        device = x.device
        
        # Initialize aggregated messages
        local_agg = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)
        medium_agg = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)
        long_agg = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=x.dtype)
        
        # Process each edge type
        for edge_type, edge_slice in edge_slices.items():
            if edge_slice.stop - edge_slice.start == 0:
                continue
            
            # Get edges and attributes for this type
            edges = edge_index[:, edge_slice]
            attrs = edge_attr[edge_slice]
            
            # Compute messages
            source = x[edges[0]]
            target = x[edges[1]]
            message_input = torch.cat([source, target, attrs], dim=-1)
            
            if edge_type == 'local':
                messages = self.local_message(message_input)
                local_agg = local_agg.index_add(0, edges[1], messages)
            elif edge_type == 'medium':
                messages = self.medium_message(message_input)
                medium_agg = medium_agg.index_add(0, edges[1], messages)
            elif edge_type == 'long':
                messages = self.long_message(message_input)
                long_agg = long_agg.index_add(0, edges[1], messages)
        
        # Combine all messages with original features
        update_input = torch.cat([x, local_agg, medium_agg, long_agg], dim=-1)
        
        # Gated update
        gate = self.update_gate(update_input)
        update = self.update_mlp(update_input)
        
        output = gate * update + (1 - gate) * x
        return self.layer_norm(output)

class CircularLoss(nn.Module):
    """Loss function with circular handling for MWD"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Split predictions and targets
        pred_swh = predictions[:, :, 0]
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        pred_mwp = predictions[:, :, 3]
        
        true_swh = targets[:, :, 0]
        true_mwd_cos = targets[:, :, 1]
        true_mwd_sin = targets[:, :, 2]
        true_mwp = targets[:, :, 3]
        
        # Standard MSE for SWH and MWP
        swh_loss = F.mse_loss(pred_swh, true_swh)
        mwp_loss = F.mse_loss(pred_mwp, true_mwp)
        
        # Circular loss for MWD
        mwd_cos_loss = F.mse_loss(pred_mwd_cos, true_mwd_cos)
        mwd_sin_loss = F.mse_loss(pred_mwd_sin, true_mwd_sin)
        mwd_loss = mwd_cos_loss + mwd_sin_loss
        
        # Total loss
        total_loss = swh_loss + mwd_loss + mwp_loss
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_loss': mwd_loss,
            'mwp_loss': mwp_loss
        }

# ==============================================================================
# MODEL V5 WITH CUBOID ATTENTION
# ==============================================================================

class MultiscaleGlobalWaveGNN_V5(nn.Module):
    """Global spatiotemporal GNN with graph-aware cuboid attention"""
    
    def __init__(self, config: HybridSamplingConfig, mesh):
        super().__init__()
        self.config = config
        self.mesh = mesh
        
        # Create graph partitioners
        if config.use_cuboid_attention:
            print("🎯 Initializing graph partitioners for cuboid attention...")
            self.fine_partitioner = GraphPartitioner(
                mesh, config.num_spatial_patches, config.patch_overlap_hops
            )
            self.coarse_partitioner = GraphPartitioner(
                mesh, config.num_coarse_patches, config.patch_overlap_hops
            )
        else:
            self.fine_partitioner = None
            self.coarse_partitioner = None
        
        # Feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.num_input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Spatial layers with multiscale message passing
        self.spatial_layers = nn.ModuleList([
            MultiscaleMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Cuboid attention (replaces full attention)
        if config.use_cuboid_attention:
            self.spatial_attention = HierarchicalCuboidAttention(
                config.hidden_dim,
                config.num_attention_heads,
                self.fine_partitioner,
                self.coarse_partitioner
            )
        else:
            # Fallback to no attention
            self.spatial_attention = nn.Identity()
        
        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.dropout if config.num_temporal_layers > 1 else 0
        )
        
        # Output heads
        self.output_mlp = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_output_features)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                edge_slices: Dict[str, slice]) -> torch.Tensor:
        batch_size, seq_len, num_nodes, num_features = x.size()
        
        # Process each timestep spatially
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, features]
            h_t = self.feature_encoder(x_t)
            
            # Spatial processing for each sample in batch
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]  # [nodes, hidden_dim]
                
                # Multiscale message passing
                for i, layer in enumerate(self.spatial_layers):
                    h_b = layer(h_b, edge_index, edge_attr, edge_slices)
                    
                    # Apply cuboid attention every 2 layers
                    if i % 2 == 1:
                        h_b = self.spatial_attention(h_b)
                
                batch_outputs.append(h_b)
            
            h_t = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(h_t)
        
        # Stack temporal sequence
        spatial_sequence = torch.stack(spatial_outputs, dim=1)  # [batch, seq_len, nodes, hidden]
        
        # Temporal processing for each node
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]  # [batch, seq_len, hidden]
            
            # LSTM
            lstm_out, _ = self.temporal_encoder(node_sequence)
            
            # Use last timestep
            final_state = lstm_out[:, -1, :]  # [batch, temporal_hidden]
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)  # [batch, nodes, temporal_hidden]
        
        # Generate predictions
        predictions = self.output_mlp(temporal_features)  # [batch, nodes, 4]
        
        return predictions

# ==============================================================================
# TRAINING WITH V5
# ==============================================================================

class HybridSamplingTrainer_V5:
    """Trainer for V5 model with cuboid attention"""
    
    def __init__(self, config: HybridSamplingConfig):
        self.config = config
        
        # Device setup
        try:
            self.device = torch.device(config.device)
            test_tensor = torch.zeros(1).to(self.device)
            del test_tensor
        except Exception as e:
            if config.use_cpu_fallback:
                print(f"⚠️  Failed to use {config.device}, falling back to CPU: {e}")
                self.device = torch.device("cpu")
            else:
                raise
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalizers
        self.feature_normalizer = RobustScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_swh_loss': [],
            'val_mwd_loss': [],
            'val_mwp_loss': [],
            'epoch_times': [],
            'samples_per_epoch': []
        }
        
        print(f"🌍 V5 Global Wave Trainer with Cuboid Attention initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup dataset with hybrid sampling"""
        print("\n📊 Setting up hybrid sampling dataset...")
        
        # Find all data files
        data_files = sorted(glob.glob(self.config.data_pattern))
        if not data_files:
            raise ValueError(f"No data files found matching pattern: {self.config.data_pattern}")
        
        print(f"   Found {len(data_files)} data files")
        
        # Create multiscale mesh
        self.mesh = MultiscaleGlobalIcosahedralMesh(
            refinement_level=self.config.mesh_refinement_level,
            config=self.config,
            cache_dir=self.config.cache_dir
        )
        
        # Get combined edge connectivity
        self.edge_index, self.edge_attr, self.edge_slices = self.mesh.get_combined_edge_index_and_attr()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        
        # Create dataset with bathymetry support
        self.dataset = BathymetryEnabledHybridSamplingDataset(
            data_paths=data_files,
            mesh=self.mesh,
            config=self.config,
            gebco_path="data/gebco/GEBCO_2023.nc"
        )
        
        # Print ocean statistics
        if hasattr(self.dataset, 'get_ocean_statistics'):
            ocean_stats = self.dataset.get_ocean_statistics()
            print(f"\n🌊 Ocean coverage statistics:")
            for key, value in ocean_stats.items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
        
        # Fit normalizers
        print("\n🔧 Fitting normalizers...")
        self._fit_normalizers(self.dataset)
        
        # Create validation dataset with fixed sampling
        print("   Creating validation split...")
        val_size = int(self.config.validation_split * self.config.samples_per_epoch)
        train_size = self.config.samples_per_epoch - val_size
        
        # For validation, we'll use a fixed subset
        self.val_indices = list(range(0, len(self.dataset), len(self.dataset) // val_size))[:val_size]
        
        # Initialize epoch samples
        print("   Initializing epoch samples...")
        if hasattr(self.dataset, '_resample_epoch'):
            self.dataset._resample_epoch()
            print(f"   ✓ Epoch samples ready: {len(self.dataset.epoch_samples)} samples")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.dataset,
            batch_sampler=DiverseBatchSampler(self.dataset, self.config.batch_size),
            num_workers=0,  # Reduced for memory
            pin_memory=True
        )
        
        # Validation uses simple sequential sampling
        self.val_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=self.val_indices,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"\n✅ Data setup complete:")
        print(f"   Mesh nodes: {len(self.mesh.vertices)}")
        print(f"   Total edges: {self.edge_index.shape[1]}")
        print(f"   Samples per epoch: {self.config.samples_per_epoch}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation samples: {len(self.val_indices)}")
        
        return self.dataset
    
    def _fit_normalizers(self, dataset):
        """Fit normalizers on sample data"""
        sample_features = []
        sample_targets = []
        
        print("   Collecting samples for normalizer fitting...")
        
        # Sample across different months
        n_samples = min(200, len(dataset))
        sample_indices = np.linspace(0, len(dataset)-1, n_samples, dtype=int)
        
        skipped_samples = 0
        for idx in sample_indices:
            try:
                sample = dataset[idx]
                features = sample['input'].numpy()
                targets = sample['target'].numpy()
                
                # Check for NaN
                if np.isnan(features).any() or np.isnan(targets).any():
                    skipped_samples += 1
                    continue
                
                features_flat = features.reshape(-1, features.shape[-1])
                sample_features.append(features_flat)
                sample_targets.append(targets)
            except Exception as e:
                print(f"   Warning: Failed to load sample {idx}: {e}")
                skipped_samples += 1
                continue
        
        if not sample_features:
            raise ValueError("Failed to load any samples for normalizer fitting")
        
        if skipped_samples > 0:
            print(f"   Skipped {skipped_samples} samples due to NaN values")
        
        all_features = np.vstack(sample_features)
        all_targets = np.vstack(sample_targets)
        
        # Replace NaN with zeros before fitting
        all_features_clean = np.nan_to_num(all_features, nan=0.0)
        all_targets_clean = np.nan_to_num(all_targets, nan=0.0)
        
        self.feature_normalizer.fit(all_features_clean)
        self.target_normalizer.fit(all_targets_clean)
        
        print(f"   ✅ Normalizers fitted on {len(all_features)} samples")
    
    def train_epoch(self, model, optimizer, criterion):
        """Train one epoch"""
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            inputs_norm = self.feature_normalizer.transform(inputs_flat)
            inputs_norm = np.nan_to_num(inputs_norm, nan=0.0, posinf=10.0, neginf=-10.0)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets
            targets_flat = targets.view(-1, 3).cpu().numpy()
            targets_norm = self.target_normalizer.transform_targets(targets_flat)
            targets_norm = np.nan_to_num(targets_norm, nan=0.0, posinf=10.0, neginf=-10.0)
            targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
            targets = targets.view(batch_size, num_nodes, 4)
            
            # Forward pass
            predictions = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ⚠️  Invalid loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            
            # Update weights
            optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}")
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate(self, model, criterion):
        """Validate model"""
        model.eval()
        val_losses = []
        val_losses_by_var = {'swh': [], 'mwd': [], 'mwp': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Normalize inputs
                batch_size, seq_len, num_nodes, num_features = inputs.size()
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.feature_normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets
                targets_flat = targets.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
                
                # Forward pass
                predictions = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                val_losses.append(loss_dict['total_loss'].item())
                
                # Store per-variable losses
                for var in ['swh', 'mwd', 'mwp']:
                    if f'{var}_loss' in loss_dict:
                        val_losses_by_var[var].append(loss_dict[f'{var}_loss'].item())
        
        mean_total_loss = np.mean(val_losses) if val_losses else float('inf')
        mean_var_losses = {
            var: np.mean(losses) if losses else 0.0 
            for var, losses in val_losses_by_var.items()
        }
        
        return mean_total_loss, mean_var_losses
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting V5 global wave model training...")
        print("✨ Key improvements over V3:")
        print("   - Graph-aware cuboid attention (100x memory reduction)")
        print("   - Hierarchical patches (local + global)")
        print("   - Batch size increased to 8")
        print("   - Expected epoch time: 1-2 hours (vs 8 hours)")
        
        # Setup data
        dataset = self.setup_data()
        
        # Create model
        model = MultiscaleGlobalWaveGNN_V5(self.config, self.mesh).to(self.device)
        print(f"\n✅ Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Using cuboid attention: {self.config.use_cuboid_attention}")
        
        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.base_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Add learning rate warmup for stability
        from torch.optim.lr_scheduler import LinearLR
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=5  # Warmup for 5 epochs
        )
        
        criterion = CircularLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n📈 Training for {self.config.num_epochs} epochs...")
        print(f"   Each epoch: {self.config.samples_per_epoch} samples")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.base_learning_rate} (with warmup)")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Resample data for new epoch
            if epoch > 0:
                dataset.on_epoch_end()
            
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion)
            
            # Validate
            val_loss, val_var_losses = self.validate(model, criterion)
            
            # Update learning rate during warmup
            if epoch < 5:
                warmup_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Learning rate: {current_lr:.2e} (warmup)")
            
            # Track time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_swh_loss'].append(val_var_losses['swh'])
            self.history['val_mwd_loss'].append(val_var_losses['mwd'])
            self.history['val_mwp_loss'].append(val_var_losses['mwp'])
            self.history['epoch_times'].append(epoch_time)
            self.history['samples_per_epoch'].append(len(dataset))
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Loss by variable: SWH={val_var_losses['swh']:.4f}, "
                  f"MWD={val_var_losses['mwd']:.4f}, MWP={val_var_losses['mwp']:.4f}")
            print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            
            # Memory usage tracking
            if self.device.type == 'mps':
                allocated = torch.mps.current_allocated_memory() / 1024**3
                print(f"   Memory: {allocated:.1f} GB")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(model, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(model, epoch, val_loss, is_best=False)
        
        # Save final model
        self.save_final_model(model)
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Total training time: {sum(self.history['epoch_times'])/60:.1f} minutes")
        print(f"   Average epoch time: {np.mean(self.history['epoch_times'])/60:.1f} minutes")
    
    def save_checkpoint(self, model, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'val_loss': val_loss,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'edge_index': self.edge_index.cpu(),
            'edge_attr': self.edge_attr.cpu(),
            'edge_slices': self.edge_slices,
            'mesh_vertices': self.mesh.vertices,
            'multiscale_edges': self.mesh.multiscale_edges,
            'history': self.history
        }
        
        if is_best:
            path = self.output_dir / "best_model_v5.pt"
        else:
            path = self.output_dir / f"checkpoint_v5_epoch_{epoch+1}.pt"
        
        torch.save(checkpoint, path)
        print(f"   💾 Saved: {path.name}")
    
    def save_final_model(self, model):
        """Save final model with metadata"""
        final_data = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'edge_index': self.edge_index.cpu(),
            'edge_attr': self.edge_attr.cpu(),
            'edge_slices': self.edge_slices,
            'mesh': {
                'vertices': self.mesh.vertices,
                'faces': self.mesh.faces,
                'refinement_level': self.mesh.refinement_level,
                'multiscale_edges': self.mesh.multiscale_edges,
                'edge_attributes': self.mesh.edge_attributes
            },
            'training_history': self.history,
            'timestamp': datetime.now().isoformat(),
            'version': 'v5_cuboid_attention'
        }
        
        path = self.output_dir / "global_wave_model_v5_final.pt"
        torch.save(final_data, path)
        
        # Also save config as JSON
        config_dict = {k: v for k, v in self.config.__dict__.items() if isinstance(v, (int, float, str, list, tuple))}
        with open(self.output_dir / "config_v5.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Final V5 model saved to: {path}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Per-variable validation losses
        ax2.plot(epochs, self.history['val_swh_loss'], 'g-', label='SWH')
        ax2.plot(epochs, self.history['val_mwd_loss'], 'b-', label='MWD')
        ax2.plot(epochs, self.history['val_mwp_loss'], 'r-', label='MWP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss by Variable')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time plot
        ax3.plot(epochs, self.history['epoch_times'], 'g-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Training Time per Epoch')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        summary_text = f"V5 Model Performance:\n\n"
        summary_text += f"Total Val Loss: {self.history['val_loss'][-1]:.4f}\n"
        summary_text += f"SWH Val Loss: {self.history['val_swh_loss'][-1]:.4f}\n"
        summary_text += f"MWD Val Loss: {self.history['val_mwd_loss'][-1]:.4f}\n"
        summary_text += f"MWP Val Loss: {self.history['val_mwp_loss'][-1]:.4f}\n\n"
        summary_text += f"Total epochs: {len(epochs)}\n"
        summary_text += f"Avg time/epoch: {np.mean(self.history['epoch_times']):.1f}s\n"
        summary_text += f"Samples/epoch: {self.config.samples_per_epoch}\n\n"
        summary_text += f"Key improvements:\n"
        summary_text += f"• Cuboid attention (100x memory ↓)\n"
        summary_text += f"• Batch size: 8 (4x ↑)\n"
        summary_text += f"• Epoch time: ~{np.mean(self.history['epoch_times'])/60:.0f} min"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontfamily='monospace', verticalalignment='top', fontsize=10)
        ax4.set_title('V5 Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history_v5.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Training history plot saved")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("🌍 GLOBAL WAVE PREDICTION MODEL V5 - CUBOID ATTENTION")
    print("=" * 70)
    print("Memory-efficient attention for global wave modeling")
    print("Complete implementation with bathymetry support")
    print("Designed for Apple M4 Pro with 128GB RAM")
    print("=" * 70)
    
    # Configuration
    config = HybridSamplingConfig()
    
    # V5 specific settings
    config.use_cuboid_attention = True
    config.num_spatial_patches = 100
    config.num_coarse_patches = 10
    config.batch_size = 8  # Increased from 2
    config.accumulation_steps = 1  # No accumulation needed
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Train V5 global wave model')
    parser.add_argument('--data-pattern', type=str, default=config.data_pattern,
                        help='Glob pattern for data files')
    parser.add_argument('--samples-per-epoch', type=int, default=config.samples_per_epoch,
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.num_epochs,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default=config.device,
                        choices=['mps', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.data_pattern = args.data_pattern
    config.samples_per_epoch = args.samples_per_epoch
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.device = args.device
    
    print(f"\n📋 V5 Configuration:")
    print(f"   Data pattern: {config.data_pattern}")
    print(f"   Samples per epoch: {config.samples_per_epoch}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Device: {config.device}")
    
    print(f"\n🔧 Model configuration:")
    print(f"   Mesh refinement: Level {config.mesh_refinement_level} (~40k nodes)")
    print(f"   Cuboid attention: ENABLED")
    print(f"   - Fine patches: {config.num_spatial_patches}")
    print(f"   - Coarse patches: {config.num_coarse_patches}")
    print(f"   - Overlap hops: {config.patch_overlap_hops}")
    print(f"   Multiscale edges: {config.use_multiscale_edges}")
    print(f"   Hidden dim: {config.hidden_dim}")
    
    print(f"\n💾 Memory optimization:")
    print(f"   V3 attention: O(40,000²) = 1.6B parameters")
    print(f"   V5 attention: O(40,000×400) = 16M parameters")
    print(f"   Memory reduction: ~100x")
    print(f"   Expected memory usage: ~30GB (vs 130GB)")
    
    # Create trainer and start training
    trainer = HybridSamplingTrainer_V5(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving current state...")
        
        # Save partial results
        if hasattr(trainer, 'history') and trainer.history['train_loss']:
            with open(trainer.output_dir / "partial_history_v5.json", 'w') as f:
                json.dump(trainer.history, f)
            print("   Saved partial history")
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        if hasattr(trainer, 'history') and trainer.history['train_loss']:
            print("\n💾 Saving partial results...")
            with open(trainer.output_dir / "partial_history_v5.json", 'w') as f:
                json.dump(trainer.history, f)
    
    print("\n🎉 V5 global wave model training complete!")
    print(f"   Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    test = torch.optim.AdamW
    main()