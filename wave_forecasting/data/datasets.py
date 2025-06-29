# data/datasets.py
"""PyTorch datasets for different training modes"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np

from config.base import DataConfig, ExperimentConfig
from data.preprocessing import clean_features_for_training, MultiResolutionInterpolator
from mesh.icosahedral import IcosahedralMesh

class MeshDataLoader:
    """Connects mesh to multi-resolution data"""
    
    def __init__(self, mesh: IcosahedralMesh, interpolator: MultiResolutionInterpolator,
                 config: DataConfig):
        self.mesh = mesh
        self.interpolator = interpolator
        self.config = config
        
        # Get regional node indices
        self.region_indices = mesh.filter_region(config.lat_bounds, config.lon_bounds)
        mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
        self.region_lats = mesh_lats[self.region_indices]
        self.region_lons = mesh_lons[self.region_indices]
        
        print(f"Mesh data loader ready: {len(self.region_indices)} regional nodes")
    
    def load_features(self, time_idx: int = 0) -> Dict[str, any]:
        """Load all features onto mesh nodes"""
        
        # Interpolate all variables to mesh nodes
        interpolated_data = self.interpolator.interpolate_to_points(
            self.region_lats, self.region_lons, time_idx=time_idx
        )
        
        # Convert to feature matrix
        feature_names = list(interpolated_data.keys())
        n_nodes = len(self.region_lats)
        n_features = len(feature_names)
        
        feature_matrix = np.zeros((n_nodes, n_features))
        for i, feature_name in enumerate(feature_names):
            feature_matrix[:, i] = interpolated_data[feature_name].ravel()
        
        return {
            'features': feature_matrix,
            'feature_names': feature_names,
            'node_indices': self.region_indices,
            'coordinates': np.column_stack([self.region_lats, self.region_lons])
        }

class SpatialWaveDataset(Dataset):
    """Dataset for single-timestep wave prediction"""
    
    def __init__(self, mesh_loader: MeshDataLoader, num_timesteps: int = 50):
        self.mesh_loader = mesh_loader
        self.samples = self._create_samples(num_timesteps)
        
    def _create_samples(self, num_timesteps: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create training samples: predict t+1 from t"""
        
        print(f"Creating spatial training data from {num_timesteps} timesteps...")
        
        samples = []
        for t in range(num_timesteps - 1):
            try:
                # Input features at time t
                input_data = self.mesh_loader.load_features(time_idx=t)
                input_features = clean_features_for_training(
                    torch.tensor(input_data['features'], dtype=torch.float32)
                )
                
                # Target wave variables at time t+1
                target_data = self.mesh_loader.load_features(time_idx=t+1)
                target_features_raw = torch.tensor(target_data['features'], dtype=torch.float32)
                
                # Extract wave variables [swh, mwd, mwp]
                feature_names = input_data['feature_names']
                wave_indices = [i for i, name in enumerate(feature_names) 
                               if name in ['swh', 'mwd', 'mwp']]
                
                target_waves = target_features_raw[:, wave_indices]
                target_waves_clean = clean_features_for_training(target_waves)
                
                samples.append((input_features, target_waves_clean))
                
            except Exception as e:
                print(f"⚠️  Skipping timestep {t}: {e}")
                continue
        
        print(f"✅ Created {len(samples)} spatial training samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_features, target_waves = self.samples[idx]
        return {
            'features': input_features,
            'targets': target_waves
        }

class TemporalWaveDataset(Dataset):
    """Dataset for temporal sequence prediction"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: ExperimentConfig,
                 max_samples: Optional[int] = None):
        self.mesh_loader = mesh_loader
        self.config = config
        self.sequences = self._create_sequences(max_samples)
    
    def _create_sequences(self, max_samples: Optional[int]) -> List[int]:
        """Generate valid sequence starting indices"""
        
        # Get total available timesteps (simplified - would need proper time management)
        total_timesteps = 200  # Placeholder - implement proper time counting
        
        sequence_length = self.config.sequence_length
        forecast_offset = self.config.forecast_horizon // self.config.data.time_step_hours
        
        max_start_idx = total_timesteps - sequence_length - forecast_offset
        sequences = list(range(0, max_start_idx, 6))  # Non-overlapping sequences
        
        if max_samples:
            sequences = sequences[:max_samples]
        
        print(f"✅ Created {len(sequences)} temporal sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        
        # Load input sequence
        input_sequence = []
        for t in range(start_idx, start_idx + self.config.sequence_length):
            features_data = self.mesh_loader.load_features(time_idx=t)
            features = clean_features_for_training(
                torch.tensor(features_data['features'], dtype=torch.float32)
            )
            input_sequence.append(features)
        
        # Load target (forecast horizon ahead)
        forecast_offset = self.config.forecast_horizon // self.config.data.time_step_hours
        target_idx = start_idx + self.config.sequence_length + forecast_offset - 1
        target_data = self.mesh_loader.load_features(time_idx=target_idx)
        
        # Extract wave variables as targets
        feature_names = target_data['feature_names']
        wave_indices = [i for i, name in enumerate(feature_names) 
                       if name in ['swh', 'mwd', 'mwp']]
        
        target_features = torch.tensor(target_data['features'], dtype=torch.float32)
        target_waves = clean_features_for_training(target_features[:, wave_indices])
        
        return {
            'input_sequence': torch.stack(input_sequence),  # [seq_len, nodes, features]
            'target': target_waves,  # [nodes, 3]
            'forecast_horizon': self.config.forecast_horizon
        }
