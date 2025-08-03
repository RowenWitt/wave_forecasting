#!/usr/bin/env python3
"""
Global Wave Model V4 - Fast Training with Pre-interpolated Zarr Data
Building on V3 architecture but with 150x faster data loading
"""

import os
import sys
import json
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import xarray as xr
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GlobalWaveConfigV4:
    """Configuration for V4 model - optimized for pre-interpolated data"""
    
    # Model architecture (proven from V3)
    mesh_refinement_level: int = 6
    n_lat_lon_features: int = 2
    n_static_features: int = 3
    n_time_features: int = 11
    hidden_dim: int = 512
    latent_dim: int = 128
    n_message_layers: int = 6
    edge_dropout: float = 0.1
    node_dropout: float = 0.15
    
    # Mesh parameters (for compatibility with V3 mesh)
    max_edge_distance_km: float = 300.0
    medium_edge_distance_km: float = 600.0
    long_edge_distance_km: float = 1200.0
    cache_dir: str = "cache/global_mesh_v4"
    use_mesh_cache: bool = True
    
    # Sequence configuration
    sequence_length: int = 4  # 24 hours of context
    
    # Target features
    target_features: List[str] = None
    input_features: List[str] = None
    
    # Training parameters
    batch_size: int = 8  # Can go higher with fast loading
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine' or 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Training schedule
    num_epochs: int = 200  # Now feasible!
    samples_per_epoch: int = 2000
    val_samples: int = 400
    
    # Loss weights
    swh_weight: float = 1.0
    mwd_weight: float = 1.0
    mwp_weight: float = 1.0
    
    # Experiment tracking
    experiment_name: str = "global_wave_v4"
    checkpoint_dir: str = "experiments/global_wave_v4"
    
    def __post_init__(self):
        self.target_features = ['swh', 'mwd', 'mwp']
        self.input_features = [
            'swh', 'mwd', 'mwp', 'shww', 'u10', 'v10', 'msl', 
            'sst', 'z_500', 't_850', 'tcwv', 'ocean_depth',
            'lat', 'lon', 'hour_sin', 'hour_cos'
        ]
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET - FAST PRE-INTERPOLATED VERSION
# ============================================================================

# class PreInterpolatedZarrDataset(Dataset):
#     """Memory-efficient dataset using lazy loading of zarr files"""
    
#     def __init__(self, 
#                  zarr_dir: str,
#                  config: GlobalWaveConfigV4,
#                  split: str = 'train',
#                  val_ratio: float = 0.2):
        
#         self.zarr_dir = Path(zarr_dir)
#         self.config = config
#         self.split = split
        
#         # Find all zarr files
#         self.zarr_paths = sorted(list(self.zarr_dir.glob("*.zarr")))
#         if not self.zarr_paths:
#             raise ValueError(f"No Zarr files found in {zarr_dir}")
        
#         print(f"\n📊 Initializing {split} dataset...")
#         print(f"   Found {len(self.zarr_paths)} Zarr files")
        
#         # Load metadata
#         metadata_path = self.zarr_dir / f"{self.zarr_paths[0].stem}_metadata.json"
#         if metadata_path.exists():
#             with open(metadata_path, 'r') as f:
#                 self.metadata = json.load(f)
#         else:
#             self.metadata = {}
        
#         # Get mesh info from first file (then close it)
#         with xr.open_zarr(self.zarr_paths[0]) as ds:
#             self.n_nodes = len(ds.node)
#             # Store coordinate data that doesn't change
#             self.lat_values = ds.lat.values / 90.0
#             self.lon_values = ds.lon.values / 180.0
#             # Get ocean depth from first timestep
#             if 'ocean_depth' in ds.data_vars:
#                 self.ocean_depth_values = ds.ocean_depth.isel(valid_time=0).values
#             else:
#                 self.ocean_depth_values = np.zeros(self.n_nodes)
        
#         print(f"   Mesh nodes: {self.n_nodes:,}")
        
#         # Analyze sequences
#         self._analyze_sequences()
        
#         # Create train/val split
#         self._create_split(val_ratio)
        
#         # Initialize normalizers
#         self.feature_normalizer = None
#         self.target_normalizer = None
        
#         # DO NOT keep files open - load on demand
#         self._current_file = None
#         self._current_ds = None
    
#     def _analyze_sequences(self):
#         """Count available sequences in each file"""
#         self.file_info = []
#         self.total_sequences = 0
        
#         for zarr_path in self.zarr_paths:
#             # Open temporarily just for counting
#             with xr.open_zarr(zarr_path) as ds:
#                 n_timesteps = len(ds.valid_time)
#                 n_sequences = n_timesteps - self.config.sequence_length
                
#                 if n_sequences > 0:
#                     self.file_info.append({
#                         'path': zarr_path,
#                         'n_timesteps': n_timesteps,
#                         'n_sequences': n_sequences,
#                         'start_idx': self.total_sequences
#                     })
#                     self.total_sequences += n_sequences
        
#         print(f"   Total sequences: {self.total_sequences:,}")
    
#     def _create_split(self, val_ratio):
#         """Create train/validation split"""
#         n_val_files = max(1, int(len(self.file_info) * val_ratio))
        
#         if self.split == 'train':
#             self.active_files = self.file_info[:-n_val_files]
#         else:
#             self.active_files = self.file_info[-n_val_files:]
        
#         self.split_sequences = sum(f['n_sequences'] for f in self.active_files)
        
#         if self.split == 'train':
#             self.samples_per_epoch = min(self.config.samples_per_epoch, self.split_sequences)
#         else:
#             self.samples_per_epoch = min(self.config.val_samples, self.split_sequences)
        
#         self._resample_epoch()
        
#         print(f"   {self.split} sequences: {self.split_sequences:,}")
#         print(f"   Samples per epoch: {self.samples_per_epoch}")
    
#     def _resample_epoch(self):
#         """Sample sequences for current epoch"""
#         valid_indices = []
#         for file_info in self.active_files:
#             for i in range(file_info['n_sequences']):
#                 valid_indices.append({
#                     'file': file_info,
#                     'local_idx': i
#                 })
        
#         if len(valid_indices) > self.samples_per_epoch:
#             indices = np.random.choice(len(valid_indices), self.samples_per_epoch, replace=False)
#             self.epoch_samples = [valid_indices[i] for i in indices]
#         else:
#             self.epoch_samples = valid_indices
#             np.random.shuffle(self.epoch_samples)
    
#     def _get_dataset(self, file_path):
#         """Get dataset, using cache if same file"""
#         if self._current_file != str(file_path):
#             # Close previous file
#             if self._current_ds is not None:
#                 self._current_ds.close()
#             # Open new file with minimal memory usage
#             self._current_ds = xr.open_zarr(file_path, chunks={'valid_time': 1, 'node': -1})
#             self._current_file = str(file_path)
#         return self._current_ds
    
#     def __len__(self):
#         return len(self.epoch_samples)
    
#     def __getitem__(self, idx):
#         """Load a single sequence with minimal memory usage"""
#         sample_info = self.epoch_samples[idx]
#         file_info = sample_info['file']
#         local_idx = sample_info['local_idx']
        
#         # Get dataset (cached if same file)
#         ds = self._get_dataset(file_info['path'])
        
#         try:
#             # Build inputs for sequence
#             input_list = []
            
#             # Get times
#             time_slice = slice(local_idx, local_idx + self.config.sequence_length)
#             times = ds.valid_time.isel(valid_time=time_slice).values
            
#             # Process each timestep
#             for t in range(self.config.sequence_length):
#                 t_idx = local_idx + t
#                 features = []
                
#                 # Dynamic features - load one timestep at a time
#                 for feat in ['swh', 'mwd', 'mwp', 'shww', 'u10', 'v10', 'msl', 
#                            'sst', 'z_500', 'z_850', 'tcwv']:
#                     if feat == 'tcwv' and feat not in ds.data_vars:
#                         features.append(np.zeros(self.n_nodes))
#                     elif feat == 'z_850' and feat in ds.data_vars:
#                         features.append(ds[feat].isel(valid_time=t_idx).values)
#                     elif feat == 't_850':  # Handle t_850 -> z_850 mapping
#                         if 'z_850' in ds.data_vars:
#                             features.append(ds['z_850'].isel(valid_time=t_idx).values)
#                         else:
#                             features.append(np.zeros(self.n_nodes))
#                     elif feat in ds.data_vars:
#                         features.append(ds[feat].isel(valid_time=t_idx).values)
#                     else:
#                         features.append(np.zeros(self.n_nodes))
                
#                 # Static features (pre-loaded)
#                 features.extend([
#                     self.ocean_depth_values,
#                     self.lat_values,
#                     self.lon_values
#                 ])
                
#                 # Time features
#                 hour = pd.Timestamp(times[t]).hour
#                 hour_sin = np.sin(2 * np.pi * hour / 24)
#                 hour_cos = np.cos(2 * np.pi * hour / 24)
#                 features.extend([
#                     np.full(self.n_nodes, hour_sin),
#                     np.full(self.n_nodes, hour_cos)
#                 ])
                
#                 input_list.append(np.stack(features, axis=-1))
            
#             # Get target
#             target_idx = local_idx + self.config.sequence_length
#             target_list = []
#             for feat in self.config.target_features:
#                 if feat in ds.data_vars:
#                     target_list.append(ds[feat].isel(valid_time=target_idx).values)
#                 else:
#                     target_list.append(np.zeros(self.n_nodes))
            
#             inputs = np.stack(input_list, axis=0)
#             targets = np.stack(target_list, axis=-1)
            
#             return {
#                 'input': torch.from_numpy(inputs.astype(np.float32)),
#                 'target': torch.from_numpy(targets.astype(np.float32))
#             }
            
#         except Exception as e:
#             print(f"\n❌ Error in __getitem__: {str(e)}")
#             # Return zeros
#             inputs = np.zeros((self.config.sequence_length, self.n_nodes, len(self.config.input_features)), dtype=np.float32)
#             targets = np.zeros((self.n_nodes, len(self.config.target_features)), dtype=np.float32)
#             return {
#                 'input': torch.from_numpy(inputs),
#                 'target': torch.from_numpy(targets)
#             }
    
#     def fit_normalizers(self, n_samples: int = 100):  # Reduced from 200
#         """Fit normalizers with minimal memory usage"""
#         print("\n🔧 Fitting normalizers...")
        
#         # Use fewer samples for normalizer fitting
#         original_samples = self.samples_per_epoch
#         self.samples_per_epoch = min(n_samples, self.split_sequences)
#         self._resample_epoch()
        
#         # Collect samples in smaller batches
#         all_features = []
#         all_targets = []
        
#         batch_size = 5  # Process 5 samples at a time
        
#         for i in range(0, len(self), batch_size):
#             batch_end = min(i + batch_size, len(self))
            
#             if i % 25 == 0:
#                 print(f"   Processing sample {i}/{len(self)}")
#                 # Log memory usage
#                 import psutil
#                 process = psutil.Process()
#                 mem_gb = process.memory_info().rss / 1024**3
#                 print(f"   Memory usage: {mem_gb:.1f} GB")
            
#             # Process batch
#             for j in range(i, batch_end):
#                 sample = self[j]
                
#                 # Only keep a subset of data for normalization
#                 features = sample['input'].numpy()
#                 targets = sample['target'].numpy()
                
#                 # Subsample nodes to reduce memory (every 10th node)
#                 node_subsample = slice(None, None, 10)
#                 features_sub = features[:, node_subsample, :].reshape(-1, features.shape[-1])
#                 targets_sub = targets[node_subsample, :]
                
#                 all_features.append(features_sub)
#                 all_targets.append(targets_sub)
            
#             # Periodically consolidate to avoid too many small arrays
#             if len(all_features) > 20:
#                 all_features = [np.vstack(all_features)]
#                 all_targets = [np.vstack(all_targets)]
        
#         # Final consolidation
#         all_features = np.vstack(all_features)
#         all_targets = np.vstack(all_targets)
        
#         # Fit normalizers
#         self.feature_normalizer = RobustScaler()
#         self.feature_normalizer.fit(all_features)
        
#         self.target_normalizer = VariableSpecificNormalizer()
#         self.target_normalizer.fit(all_targets)
        
#         print(f"   ✅ Normalizers fitted on {all_features.shape[0]:,} samples")
        
#         # Clear memory
#         del all_features, all_targets
        
#         # Restore
#         self.samples_per_epoch = original_samples
#         self._resample_epoch()
    
#     def __del__(self):
#         """Clean up open files"""
#         if hasattr(self, '_current_ds') and self._current_ds is not None:
#             self._current_ds.close()
class PreInterpolatedZarrDataset(Dataset):
    """Fast dataset using pre-interpolated Zarr files - V3-style file handling"""
    
    def __init__(self, 
                 zarr_dir: str,
                 config: GlobalWaveConfigV4,
                 split: str = 'train',
                 val_ratio: float = 0.2):
        
        self.zarr_dir = Path(zarr_dir)
        self.config = config
        self.split = split
        
        # Find all zarr files
        self.zarr_paths = sorted(list(self.zarr_dir.glob("*.zarr")))
        if not self.zarr_paths:
            raise ValueError(f"No Zarr files found in {zarr_dir}")
        
        print(f"\n📊 Initializing {split} dataset...")
        print(f"   Found {len(self.zarr_paths)} Zarr files")
        
        # Load metadata
        metadata_path = self.zarr_dir / f"{self.zarr_paths[0].stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Get mesh info from first file (then close it immediately)
        with xr.open_zarr(self.zarr_paths[0]) as ds:
            self.n_nodes = len(ds.node)
            # Store coordinate data that doesn't change
            self.lat_values = ds.lat.values / 90.0
            self.lon_values = ds.lon.values / 180.0
            # Get ocean depth from first timestep
            if 'ocean_depth' in ds.data_vars:
                self.ocean_depth_values = ds.ocean_depth.isel(valid_time=0).values
            else:
                self.ocean_depth_values = np.zeros(self.n_nodes)
        # File is now closed!
        
        print(f"   Mesh nodes: {self.n_nodes:,}")
        
        # Analyze sequences
        self._analyze_sequences()
        
        # Create train/val split
        self._create_split(val_ratio)
        
        # Initialize normalizers
        self.feature_normalizer = None
        self.target_normalizer = None
        
        # V3-style caching - only keep one file open
        self.current_path = None
        self.current_data = None
    
    def _analyze_sequences(self):
        """Count available sequences in each file"""
        self.file_info = []
        self.total_sequences = 0
        
        for zarr_path in self.zarr_paths:
            # Open temporarily just for counting
            with xr.open_zarr(zarr_path) as ds:
                n_timesteps = len(ds.valid_time)
                n_sequences = n_timesteps - self.config.sequence_length
                
                if n_sequences > 0:
                    self.file_info.append({
                        'path': zarr_path,
                        'n_timesteps': n_timesteps,
                        'n_sequences': n_sequences,
                        'start_idx': self.total_sequences
                    })
                    self.total_sequences += n_sequences
        
        print(f"   Total sequences: {self.total_sequences:,}")
    
    def _create_split(self, val_ratio):
        """Create train/validation split"""
        n_val_files = max(1, int(len(self.file_info) * val_ratio))
        
        if self.split == 'train':
            self.active_files = self.file_info[:-n_val_files]
        else:
            self.active_files = self.file_info[-n_val_files:]
        
        self.split_sequences = sum(f['n_sequences'] for f in self.active_files)
        
        if self.split == 'train':
            self.samples_per_epoch = min(self.config.samples_per_epoch, self.split_sequences)
        else:
            self.samples_per_epoch = min(self.config.val_samples, self.split_sequences)
        
        self._resample_epoch()
        
        print(f"   {self.split} sequences: {self.split_sequences:,}")
        print(f"   Samples per epoch: {self.samples_per_epoch}")
    
    def _resample_epoch(self):
        """Sample sequences for current epoch"""
        valid_indices = []
        for file_info in self.active_files:
            for i in range(file_info['n_sequences']):
                valid_indices.append({
                    'file': file_info,
                    'local_idx': i
                })
        
        if len(valid_indices) > self.samples_per_epoch:
            indices = np.random.choice(len(valid_indices), self.samples_per_epoch, replace=False)
            self.epoch_samples = [valid_indices[i] for i in indices]
        else:
            self.epoch_samples = valid_indices
            np.random.shuffle(self.epoch_samples)
    
    def _load_data(self, path: str):
        """Load data file with caching - V3 style"""
        path_str = str(path)
        if self.current_path != path_str:
            # Close previous file if open
            if self.current_data is not None:
                self.current_data.close()
                self.current_data = None
            
            # Open new file
            self.current_data = xr.open_zarr(path)
            self.current_path = path_str
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        """Load a single sequence - V3 style"""
        sample_info = self.epoch_samples[idx]
        file_info = sample_info['file']
        local_idx = sample_info['local_idx']
        
        # Load data file if needed (V3 style caching)
        self._load_data(file_info['path'])
        
        try:
            # Build inputs for sequence
            input_list = []
            
            # Process each timestep
            for t in range(self.config.sequence_length):
                t_idx = local_idx + t
                features = []
                
                # Dynamic features - load directly from xarray
                for feat in ['swh', 'mwd', 'mwp', 'shww', 'u10', 'v10', 'msl', 
                           'sst', 'z_500', 'z_850', 'tcwv']:
                    if feat == 'tcwv' and feat not in self.current_data.data_vars:
                        features.append(np.zeros(self.n_nodes))
                    elif feat == 'z_850' and feat in self.current_data.data_vars:
                        # Use xarray's isel for efficient access
                        data = self.current_data[feat].isel(valid_time=t_idx).values
                        features.append(data)
                    elif feat == 't_850':  # Handle t_850 -> z_850 mapping
                        if 'z_850' in self.current_data.data_vars:
                            data = self.current_data['z_850'].isel(valid_time=t_idx).values
                            features.append(data)
                        else:
                            features.append(np.zeros(self.n_nodes))
                    elif feat in self.current_data.data_vars:
                        data = self.current_data[feat].isel(valid_time=t_idx).values
                        features.append(data)
                    else:
                        features.append(np.zeros(self.n_nodes))
                
                # Static features (pre-loaded)
                features.extend([
                    self.ocean_depth_values,
                    self.lat_values,
                    self.lon_values
                ])
                
                # Time features
                time_val = self.current_data.valid_time.isel(valid_time=t_idx).values
                hour = pd.Timestamp(time_val).hour
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                features.extend([
                    np.full(self.n_nodes, hour_sin),
                    np.full(self.n_nodes, hour_cos)
                ])
                
                input_list.append(np.stack(features, axis=-1))
            
            # Get target
            target_idx = local_idx + self.config.sequence_length
            target_list = []
            for feat in self.config.target_features:
                if feat in self.current_data.data_vars:
                    data = self.current_data[feat].isel(valid_time=target_idx).values
                    target_list.append(data)
                else:
                    target_list.append(np.zeros(self.n_nodes))
            
            inputs = np.stack(input_list, axis=0)
            targets = np.stack(target_list, axis=-1)
            
            return {
                'input': torch.from_numpy(inputs.astype(np.float32)),
                'target': torch.from_numpy(targets.astype(np.float32))
            }
            
        except Exception as e:
            print(f"\n❌ Error in __getitem__: {str(e)}")
            # Return zeros
            inputs = np.zeros((self.config.sequence_length, self.n_nodes, len(self.config.input_features)), dtype=np.float32)
            targets = np.zeros((self.n_nodes, len(self.config.target_features)), dtype=np.float32)
            return {
                'input': torch.from_numpy(inputs),
                'target': torch.from_numpy(targets)
            }
    
    def fit_normalizers(self, n_samples: int = 100):
        """Fit normalizers with minimal memory usage"""
        print("\n🔧 Fitting normalizers...")
        
        # Use fewer samples for normalizer fitting
        original_samples = self.samples_per_epoch
        self.samples_per_epoch = min(n_samples, self.split_sequences)
        self._resample_epoch()
        
        # Collect samples in smaller batches
        all_features = []
        all_targets = []
        
        batch_size = 5  # Process 5 samples at a time
        
        for i in range(0, len(self), batch_size):
            batch_end = min(i + batch_size, len(self))
            
            if i % 25 == 0:
                print(f"   Processing sample {i}/{len(self)}")
                # Log memory usage
                import psutil
                process = psutil.Process()
                mem_gb = process.memory_info().rss / 1024**3
                print(f"   Memory usage: {mem_gb:.1f} GB")
            
            # Process batch
            for j in range(i, batch_end):
                sample = self[j]
                
                # Only keep a subset of data for normalization
                features = sample['input'].numpy()
                targets = sample['target'].numpy()
                
                # Subsample nodes to reduce memory (every 10th node)
                node_subsample = slice(None, None, 10)
                features_sub = features[:, node_subsample, :].reshape(-1, features.shape[-1])
                targets_sub = targets[node_subsample, :]
                
                all_features.append(features_sub)
                all_targets.append(targets_sub)
            
            # Periodically consolidate to avoid too many small arrays
            if len(all_features) > 20:
                all_features = [np.vstack(all_features)]
                all_targets = [np.vstack(all_targets)]
        
        # Final consolidation
        all_features = np.vstack(all_features)
        all_targets = np.vstack(all_targets)
        
        # Fit normalizers
        self.feature_normalizer = RobustScaler()
        self.feature_normalizer.fit(all_features)
        
        self.target_normalizer = VariableSpecificNormalizer()
        self.target_normalizer.fit(all_targets)
        
        print(f"   ✅ Normalizers fitted on {all_features.shape[0]:,} samples")
        
        # Clear memory
        del all_features, all_targets
        
        # Restore
        self.samples_per_epoch = original_samples
        self._resample_epoch()
    
    def __del__(self):
        """Clean up open files"""
        if hasattr(self, 'current_data') and self.current_data is not None:
            self.current_data.close()
# ============================================================================
# MODEL ARCHITECTURE (FROM V3)
# ============================================================================

class MultiscaleEdgeLayer(nn.Module):
    """Enhanced edge processing with multiscale support"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # Edge features
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_out = self.edge_mlp(edge_features)
        
        # Aggregate messages
        aggr_out = torch.zeros_like(x)
        aggr_out = aggr_out.index_add(0, col, edge_out)
        
        # Node update
        node_features = torch.cat([x, aggr_out], dim=-1)
        node_out = self.node_mlp(node_features)
        
        # Residual connection
        out = self.layer_norm(x + self.dropout(node_out))
        
        return out

class GlobalWaveGNNV4(nn.Module):
    """V4 model - same architecture as V3 but cleaner"""
    
    def __init__(self, config: GlobalWaveConfigV4):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.input_dim = len(config.input_features)
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.node_dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            MultiscaleEdgeLayer(
                self.hidden_dim, 
                3,  # edge features: distance, lat_diff, lon_diff
                self.hidden_dim,
                config.edge_dropout
            ) for _ in range(config.n_message_layers)
        ])
        
        # Temporal processing
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.latent_dim,
            num_layers=2,
            batch_first=False,  # [seq_len, batch*nodes, hidden]
            dropout=config.node_dropout
        )
        
        # Output decoders
        self.swh_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.node_dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.mwd_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.node_dropout),
            nn.Linear(self.hidden_dim // 2, 2)  # cos, sin components
        )
        
        self.mwp_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.node_dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                edge_slices: Optional[Dict[str, slice]] = None) -> torch.Tensor:
        """
        Forward pass
        x: [batch, seq_len, nodes, features]
        Returns: [batch, nodes, 4] - [swh, mwd_cos, mwd_sin, mwp]
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Process each timestep
        temporal_features = []
        
        for t in range(seq_len):
            # Get features for this timestep
            h = x[:, t]  # [batch, nodes, features]
            
            # Encode nodes
            h = h.reshape(-1, self.input_dim)  # [batch*nodes, features]
            h = self.node_encoder(h)  # [batch*nodes, hidden]
            
            # Message passing with multiscale edges
            if edge_slices is not None:
                # Process each edge type separately
                h_multi = []
                for edge_type, edge_slice in edge_slices.items():
                    h_type = h.clone()
                    edge_idx = edge_index[:, edge_slice]
                    edge_feat = edge_attr[edge_slice]
                    
                    for layer in self.message_layers:
                        h_type = layer(h_type, edge_idx, edge_feat)
                    
                    h_multi.append(h_type)
                
                # Combine multiscale features
                h = torch.stack(h_multi, dim=0).mean(dim=0)
            else:
                # Standard message passing
                for layer in self.message_layers:
                    h = layer(h, edge_index, edge_attr)
            
            h = h.reshape(batch_size, num_nodes, -1)  # [batch, nodes, hidden]
            temporal_features.append(h)
        
        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=0)  # [seq_len, batch, nodes, hidden]
        
        # Reshape for LSTM
        temporal_features = temporal_features.reshape(seq_len, batch_size * num_nodes, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(temporal_features)  # [seq_len, batch*nodes, latent]
        
        # Take last timestep
        h = lstm_out[-1]  # [batch*nodes, latent]
        h = h.reshape(batch_size, num_nodes, -1)  # [batch, nodes, latent]
        
        # Decode outputs
        swh = self.swh_decoder(h)  # [batch, nodes, 1]
        mwd = self.mwd_decoder(h)  # [batch, nodes, 2] 
        mwp = self.mwp_decoder(h)  # [batch, nodes, 1]
        
        # Normalize MWD to unit circle
        mwd = F.normalize(mwd, p=2, dim=-1)
        
        # Concatenate outputs
        output = torch.cat([swh, mwd, mwp], dim=-1)  # [batch, nodes, 4]
        
        return output

# ============================================================================
# NORMALIZERS (FROM V3)
# ============================================================================

class CircularNormalizer:
    """Normalizer for circular variables (wave direction)"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, angles_deg: np.ndarray):
        """Fit normalizer on circular data"""
        self.fitted = True
    
    def transform(self, angles_deg: np.ndarray) -> np.ndarray:
        """Transform angles to [cos, sin] representation"""
        angles_rad = np.deg2rad(angles_deg)
        return np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])
    
    def inverse_transform(self, circular: np.ndarray) -> np.ndarray:
        """Transform [cos, sin] back to degrees"""
        angles_rad = np.arctan2(circular[:, 1], circular[:, 0])
        angles_deg = np.rad2deg(angles_rad)
        return np.where(angles_deg < 0, angles_deg + 360, angles_deg)

class VariableSpecificNormalizer:
    """Separate normalizers for each wave variable"""
    
    def __init__(self):
        self.swh_scaler = RobustScaler()
        self.mwd_normalizer = CircularNormalizer()
        self.mwp_scaler = RobustScaler()
        self.fitted = False
    
    def fit(self, targets: np.ndarray):
        """Fit normalizers on target data [N, 3]"""
        self.swh_scaler.fit(targets[:, 0:1])
        self.mwd_normalizer.fit(targets[:, 1])
        self.mwp_scaler.fit(targets[:, 2:3])
        self.fitted = True
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets to normalized space [N, 4]"""
        swh_norm = self.swh_scaler.transform(targets[:, 0:1])
        mwd_norm = self.mwd_normalizer.transform(targets[:, 1])
        mwp_norm = self.mwp_scaler.transform(targets[:, 2:3])
        return np.concatenate([swh_norm, mwd_norm, mwp_norm], axis=1)
    
    def inverse_transform_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Transform from normalized [N, 4] back to [N, 3]"""
        swh = self.swh_scaler.inverse_transform(normalized[:, 0:1])
        mwd = self.mwd_normalizer.inverse_transform(normalized[:, 1:3])
        mwp = self.mwp_scaler.inverse_transform(normalized[:, 3:4])
        return np.column_stack([swh.flatten(), mwd, mwp.flatten()])

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class WaveLoss(nn.Module):
    """Loss function with proper handling of circular variables"""
    
    def __init__(self, config: GlobalWaveConfigV4):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        predictions: [batch, nodes, 4] - [swh, mwd_cos, mwd_sin, mwp]
        targets: [batch, nodes, 4] - normalized targets
        """
        
        # Extract components
        pred_swh = predictions[:, :, 0]
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        pred_mwp = predictions[:, :, 3]
        
        true_swh = targets[:, :, 0]
        true_mwd_cos = targets[:, :, 1]
        true_mwd_sin = targets[:, :, 2]
        true_mwp = targets[:, :, 3]
        
        # MSE losses
        swh_loss = F.mse_loss(pred_swh, true_swh)
        mwp_loss = F.mse_loss(pred_mwp, true_mwp)
        
        # Circular loss for MWD
        mwd_cos_loss = F.mse_loss(pred_mwd_cos, true_mwd_cos)
        mwd_sin_loss = F.mse_loss(pred_mwd_sin, true_mwd_sin)
        mwd_loss = mwd_cos_loss + mwd_sin_loss
        
        # Weighted total
        total_loss = (self.config.swh_weight * swh_loss +
                     self.config.mwd_weight * mwd_loss +
                     self.config.mwp_weight * mwp_loss)
        
        return {
            'total': total_loss,
            'swh': swh_loss,
            'mwd': mwd_loss,
            'mwp': mwp_loss
        }

# ============================================================================
# TRAINING
# ============================================================================

# Additional modifications for memory-efficient training

class GlobalWaveTrainerV4:
    """Fast trainer for V4 model with memory optimizations"""
    
    def __init__(self, config: GlobalWaveConfigV4):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"\n🌊 Global Wave Model V4 - Fast Training")
        print(f"📍 Device: {self.device}")
        print(f"📁 Checkpoint directory: {config.checkpoint_dir}")
        
        # Setup experiment directory
        self.exp_dir = Path(config.checkpoint_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_swh_loss': [],
            'val_mwd_loss': [],
            'val_mwp_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def setup_data(self, zarr_dir: str):
        """Setup datasets with memory optimization"""
        print(f"\n📊 Setting up datasets from {zarr_dir}")
        
        # Create datasets
        self.train_dataset = PreInterpolatedZarrDataset(
            zarr_dir=zarr_dir,
            config=self.config,
            split='train'
        )
        
        self.val_dataset = PreInterpolatedZarrDataset(
            zarr_dir=zarr_dir,
            config=self.config,
            split='val'
        )
        
        # Fit normalizers with fewer samples
        self.train_dataset.fit_normalizers(n_samples=50)  # Reduced
        
        # Share normalizers
        self.val_dataset.feature_normalizer = self.train_dataset.feature_normalizer
        self.val_dataset.target_normalizer = self.train_dataset.target_normalizer
        
        # Create memory-efficient dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # No multiprocessing for now
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,  # Disable prefetching
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )
        
        print(f"   ✅ Train batches: {len(self.train_loader)}")
        print(f"   ✅ Val batches: {len(self.val_loader)}")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def setup_model(self, mesh_path: str):
        """Setup model and optimization"""
        print(f"\n🔧 Setting up model...")
        
        # Load mesh and edges
        with open(mesh_path, 'rb') as f:
            mesh_data = pickle.load(f)
        
        self.edge_index = mesh_data['edge_index'].to(self.device)
        self.edge_attr = mesh_data['edge_attr'].to(self.device)
        self.edge_slices = mesh_data.get('edge_slices', None)
        
        # Create model
        self.model = GlobalWaveGNNV4(self.config).to(self.device)
        
        # Loss function
        self.criterion = WaveLoss(self.config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        if self.config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,  # Restart every 20 epochs
                T_mult=2,
                eta_min=self.config.min_lr
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=self.config.min_lr
            )
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ Model parameters: {n_params:,}")
        print(f"   ✅ Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"   ✅ Scheduler: {self.config.scheduler_type}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with aggressive memory management"""
        self.model.train()
        total_loss = 0.0
        
        # Resample epoch data
        self.train_dataset._resample_epoch()
        
        # Force garbage collection
        import gc
        import psutil
        
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        process = psutil.Process()
        print(f"\n💾 Memory at epoch start: {process.memory_info().rss / 1024**3:.1f} GB")
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                inputs = batch['input'].to(self.device, non_blocking=True)
                print(f"\n💾 Memory after input load: {process.memory_info().rss / 1024**3:.1f} GB")

                targets = batch['target'].to(self.device, non_blocking=True)
                print(f"\n💾 Memory after target load: {process.memory_info().rss / 1024**3:.1f} GB")

                # Get shapes
                batch_size, seq_len, num_nodes, num_features = inputs.shape
                
                # Normalize inputs on device
                inputs_flat = inputs.view(-1, num_features)
                print(f"\n💾 Memory after input flatten: {process.memory_info().rss / 1024**3:.1f} GB")

                
                if not hasattr(self, '_feature_scale'):
                    self._feature_scale = torch.tensor(
                        self.train_dataset.feature_normalizer.scale_, 
                        device=self.device, dtype=torch.float32
                    )
                    self._feature_center = torch.tensor(
                        self.train_dataset.feature_normalizer.center_,
                        device=self.device, dtype=torch.float32
                    )
                
                inputs_norm = (inputs_flat - self._feature_center) / self._feature_scale
                inputs = inputs_norm.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets
                targets_flat = targets.view(-1, 3)
                targets_np = targets_flat.cpu().numpy()
                targets_norm = self.train_dataset.target_normalizer.transform_targets(targets_np)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
                print(f"\n💾 Memory after norm: {process.memory_info().rss / 1024**3:.1f} GB")

                # Forward pass
                self.optimizer.zero_grad(set_to_none=True)
                print(f"\n💾 Memory after forward: {process.memory_info().rss / 1024**3:.1f} GB")

                # Use gradient checkpointing if available
                predictions = self.model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
                print(f"\n💾 Memory after predictions: {process.memory_info().rss / 1024**3:.1f} GB")

                # Loss
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total']
                
                # Backward
                loss.backward()
                print(f"\n💾 Memory after backward: {process.memory_info().rss / 1024**3:.1f} GB")

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                # Step
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Clear tensors immediately
                del inputs, targets, predictions, inputs_norm, targets_norm
                
                # Progress and memory monitoring
                if batch_idx % 10 == 0:
                    mem_gb = process.memory_info().rss / 1024**3
                    print(f"   Batch {batch_idx}/{len(self.train_loader)}: "
                          f"loss={loss.item():.4f} "
                          f"Memory: {mem_gb:.1f} GB")
                    
                    # Aggressive cleanup every 10 batches
                    gc.collect()
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    
                    # Check if memory is getting too high
                    if mem_gb > 100:  # If over 100GB, do extra cleanup
                        print(f"   ⚠️  High memory usage, forcing cleanup...")
                        del loss, loss_dict
                        gc.collect()
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        torch.mps.synchronize()  # Wait for all operations to complete
                
            except Exception as e:
                print(f"\n❌ Error in batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try to recover
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                continue
        
        # Final cleanup
        gc.collect()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        return total_loss / max(1, len(self.train_loader))
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {'total': 0.0, 'swh': 0.0, 'mwd': 0.0, 'mwp': 0.0}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Normalize (same as training)
                batch_size, seq_len, num_nodes, num_features = inputs.shape
                
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.val_dataset.feature_normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                targets_flat = targets.view(-1, 3).cpu().numpy()
                targets_norm = self.val_dataset.target_normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
                
                # Forward pass
                predictions = self.model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
                
                # Loss
                loss_dict = self.criterion(predictions, targets)
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key].item()
        
        # Average
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'feature_normalizer': self.train_dataset.feature_normalizer,
            'target_normalizer': self.train_dataset.target_normalizer,
            'edge_index': self.edge_index.cpu(),
            'edge_attr': self.edge_attr.cpu(),
            'edge_slices': self.edge_slices,
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, self.exp_dir / 'latest_checkpoint.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'best_model.pt')
            print(f"   💾 Saved best model (val_loss={val_loss:.4f})")
    
    def train(self, zarr_dir: str, mesh_path: str, resume: bool = False):
        """Main training loop"""
        print(f"\n🚀 Starting V4 training with fast data loading!")
        
        # Setup
        self.setup_data(zarr_dir)
        self.setup_model(mesh_path)
        
        # Resume if requested
        if resume and (self.exp_dir / 'latest_checkpoint.pt').exists():
            print(f"\n📂 Resuming from checkpoint...")
            checkpoint = torch.load(self.exp_dir / 'latest_checkpoint.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.history = checkpoint['history']
            self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')
            print(f"   ✅ Resumed from epoch {self.start_epoch}")
        
        # Training loop
        print(f"\n🔄 Training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            print(f"\n📈 Epoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate()
            val_loss = val_losses['total']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_swh_loss'].append(val_losses['swh'])
            self.history['val_mwd_loss'].append(val_losses['mwd'])
            self.history['val_mwp_loss'].append(val_losses['mwp'])
            self.history['epoch_times'].append(time.time() - epoch_start)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler
            if self.config.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print summary
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f} {'*BEST*' if is_best else ''}")
            print(f"   Val Losses: SWH={val_losses['swh']:.4f}, "
                  f"MWD={val_losses['mwd']:.4f}, MWP={val_losses['mwp']:.4f}")
            print(f"   Time: {self.history['epoch_times'][-1]:.1f}s")
            print(f"   LR: {self.history['learning_rates'][-1]:.6f}")
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final plots
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component losses
        axes[0, 1].plot(self.history['val_swh_loss'], label='SWH')
        axes[0, 1].plot(self.history['val_mwd_loss'], label='MWD')
        axes[0, 1].plot(self.history['val_mwp_loss'], label='MWP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Epoch times
        axes[1, 1].plot(self.history['epoch_times'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Epoch Duration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_history.png', dpi=300)
        plt.close()

# ============================================================================
# MAIN
# ============================================================================

def prepare_mesh_file(mesh_path: str, config: GlobalWaveConfigV4):
    """Prepare mesh and edges file if not exists"""
    if Path(mesh_path).exists():
        print(f"✅ Using existing mesh file: {mesh_path}")
        return
    
    print(f"🔧 Creating mesh file...")
    
    # Import mesh creation tools
    from global_wave_model_hybrid_sampling_v3 import MultiscaleGlobalIcosahedralMesh
    
    # Create mesh
    mesh = MultiscaleGlobalIcosahedralMesh(
        refinement_level=config.mesh_refinement_level,
        config=config
    )
    
    # Get vertices for computing edges
    mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
    
    # Compute standard edges with features
    edges = []
    edge_features = []
    
    print(f"   Computing edges for {len(mesh.vertices)} vertices...")
    
    # Use the multiscale edges from mesh
    for edge_type, edge_array in mesh.multiscale_edges.items():
        print(f"   Processing {edge_type} edges: {len(edge_array)}")
        for i, j in edge_array:
            # Bidirectional edges
            edges.extend([[i, j], [j, i]])
            
            # Edge features: [distance, lat_diff, lon_diff]
            lat_diff = mesh_lats[j] - mesh_lats[i]
            lon_diff = mesh_lons[j] - mesh_lons[i]
            # Handle longitude wraparound
            if lon_diff > 180:
                lon_diff -= 360
            elif lon_diff < -180:
                lon_diff += 360
            
            # Approximate distance (simplified)
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            
            edge_feat = [distance/100.0, lat_diff/90.0, lon_diff/180.0]
            edge_features.extend([edge_feat, edge_feat])
    
    # Convert to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    # Create edge slices for different edge types
    edge_slices = {}
    current_idx = 0
    
    for edge_type, edge_array in mesh.multiscale_edges.items():
        n_edges = len(edge_array) * 2  # bidirectional
        edge_slices[edge_type] = slice(current_idx, current_idx + n_edges)
        current_idx += n_edges
    
    print(f"   Total edges: {edge_index.shape[1]}")
    
    # Save
    mesh_data = {
        'vertices': mesh.vertices,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_slices': edge_slices,
        'config': config
    }
    
    with open(mesh_path, 'wb') as f:
        pickle.dump(mesh_data, f)
    
    print(f"✅ Mesh saved to: {mesh_path}")

# Also modify the configuration to use smaller batch sizes initially
def main():
    """Main training function with aggressive memory optimization"""
    
    # Start with smaller configuration
    config = GlobalWaveConfigV4(
        experiment_name="global_wave_v4_fast",
        num_epochs=200,
        
        # Reduced memory footprint
        batch_size=1,  # Start very small
        learning_rate=1e-4,
        samples_per_epoch=500,  # Fewer samples per epoch
        val_samples=100,
        
        # Smaller model initially
        hidden_dim=256,  # Reduced from 512
        latent_dim=64,   # Reduced from 128
        n_message_layers=4,  # Reduced from 6
        
        # Other settings
        gradient_clip=1.0,
        scheduler_type='cosine',
        warmup_epochs=5,
        min_lr=1e-6,
    )
    
    # Update input features
    config.input_features = [
        'swh', 'mwd', 'mwp', 'shww', 'u10', 'v10', 'msl', 
        'sst', 'z_500', 'z_850', 'tcwv', 'ocean_depth',
        'lat', 'lon', 'hour_sin', 'hour_cos'
    ]
    
    # Paths
    zarr_dir = "data/v1_global/interpolated"
    mesh_path = "cache/mesh_v4.pkl"
    
    # Monitor memory before starting
    import psutil
    import gc
    
    process = psutil.Process()
    print(f"\n💾 Initial memory: {process.memory_info().rss / 1024**3:.1f} GB")
    
    # Aggressive garbage collection
    gc.collect()
    
    # Prepare mesh
    prepare_mesh_file(mesh_path, config)
    
    # Create trainer
    trainer = GlobalWaveTrainerV4(config)
    
    # Train
    trainer.train(
        zarr_dir=zarr_dir,
        mesh_path=mesh_path,
        resume=False
    )


if __name__ == "__main__":
    main()