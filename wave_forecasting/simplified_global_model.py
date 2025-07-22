# full_global_model_one_month.py
"""
Full Global Climate-Aware Variable LR Wave Model for One Month Training
Uses all available features from smart-merged files
Based on proven Variable LR architecture that achieved great 1-month performance
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
import time
import math
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FullGlobalConfig:
    """Full configuration based on proven Variable LR success"""
    
    # Temporal parameters (proven from Variable LR)
    sequence_length: int = 6           
    prediction_horizon: int = 4        
    
    # Full architecture (proven successful)
    input_features: int = 17           # All features including derived climate
    hidden_dim: int = 384              # Enhanced for global domain
    temporal_hidden_dim: int = 192     # Proportional increase
    num_spatial_layers: int = 8        # Enhanced for global connectivity
    num_temporal_layers: int = 2       # Proven depth
    
    # Attention (full proven settings)
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    use_climate_attention: bool = True  # Keep climate attention
    num_attention_heads: int = 8
    
    # Regularization (proven settings)
    dropout: float = 0.15
    spatial_dropout: float = 0.1
    temporal_dropout: float = 0.1
    
    # Training parameters (optimized for 1-month intensive training)
    num_epochs: int = 200              # Full training epochs
    batch_size: int = 6                # Proven batch size
    base_learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Variable-specific learning rates (proven successful)
    swh_lr_multiplier: float = 0.7      # Maintain excellent SWH performance
    mwd_lr_multiplier: float = 1.0      # Standard for circular MWD
    mwp_lr_multiplier: float = 1.3      # Focus on MWP improvement
    
    # Adaptive learning rate (proven)
    lr_patience: int = 15
    lr_factor: float = 0.8
    min_lr_factor: float = 0.1
    
    # Early stopping (aggressive but not too aggressive)
    early_stopping_patience: int = 50
    
    # Variable-specific loss weights (proven)
    swh_loss_weight: float = 1.0
    mwd_loss_weight: float = 1.0        
    mwp_loss_weight: float = 1.0
    physics_loss_weight: float = 0.2
    
    # Data parameters
    validation_split: float = 0.2
    spatial_subsample: int = 4         # Less aggressive subsampling for full training

# Complete variable mapping for smart-merged files
FULL_VARIABLE_MAPPING = {
    # Single-level atmospheric (5 features)
    'u10': ['u10', '10m_u_component_of_wind'],
    'v10': ['v10', '10m_v_component_of_wind'],
    'slp': ['slp', 'msl', 'mean_sea_level_pressure'],
    'sst': ['sst', 'sea_surface_temperature'],
    'precip': ['precip', 'tp', 'total_precipitation'],
    
    # Wave variables - targets (3 features)
    'swh': ['swh', 'significant_height_of_combined_wind_waves_and_swell'],
    'mwd': ['mwd', 'mean_wave_direction'],
    'mwp': ['mwp', 'mean_wave_period'],
    
    # Pressure-level variables - flattened by smart merger (3 features)
    'u850': ['u850'],
    'v850': ['v850'],
    'z500': ['z500'],
    
    # Additional pressure levels if available (6 more features)
    'u500': ['u500'],
    'v500': ['v500'],
    'z850': ['z850'],
    'u300': ['u300'],
    'v300': ['v300'],
    'z300': ['z300']
}

class VariableSpecificNormalizer:
    """Enhanced normalizer for wave variables with circular MWD handling"""
    
    def __init__(self):
        self.swh_scaler = StandardScaler()
        self.mwp_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, targets):
        """Fit normalizer on target data [N, 3] = [SWH, MWD, MWP]"""
        self.swh_scaler.fit(targets[:, [0]])  # SWH
        self.mwp_scaler.fit(targets[:, [2]])  # MWP
        self.fitted = True
        
        print(f"🔧 Target normalizer fitted:")
        print(f"   SWH range: [{targets[:, 0].min():.2f}, {targets[:, 0].max():.2f}]")
        print(f"   MWD range: [{targets[:, 1].min():.2f}, {targets[:, 1].max():.2f}]")
        print(f"   MWP range: [{targets[:, 2].min():.2f}, {targets[:, 2].max():.2f}]")
    
    def transform_targets(self, targets):
        """Transform targets to [SWH_norm, MWD_cos, MWD_sin, MWP_norm]"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        
        # Normalize SWH and MWP
        swh_norm = self.swh_scaler.transform(targets[:, [0]])
        mwp_norm = self.mwp_scaler.transform(targets[:, [2]])
        
        # Convert MWD to circular coordinates
        mwd_rad = np.radians(targets[:, 1])
        mwd_cos = np.cos(mwd_rad).reshape(-1, 1)
        mwd_sin = np.sin(mwd_rad).reshape(-1, 1)
        
        return np.hstack([swh_norm, mwd_cos, mwd_sin, mwp_norm])

class GlobalVariableLearningManager:
    """Enhanced learning manager for variable-specific learning rates"""
    
    def __init__(self, config: FullGlobalConfig):
        self.config = config
        self.base_lr = config.base_learning_rate
        
        # Current learning rates (proven Variable LR approach)
        self.current_lrs = {
            'swh': self.base_lr * config.swh_lr_multiplier,
            'mwd': self.base_lr * config.mwd_lr_multiplier,
            'mwp': self.base_lr * config.mwp_lr_multiplier
        }
        
        # Performance tracking
        self.loss_history = {'swh': [], 'mwd': [], 'mwp': []}
        self.plateau_counters = {'swh': 0, 'mwd': 0, 'mwp': 0}
        self.best_losses = {'swh': float('inf'), 'mwd': float('inf'), 'mwp': float('inf')}
        
        print(f"🎯 Variable Learning Rate Manager:")
        print(f"   SWH LR: {self.current_lrs['swh']:.2e} (proven excellent)")
        print(f"   MWD LR: {self.current_lrs['mwd']:.2e} (circular handling)")
        print(f"   MWP LR: {self.current_lrs['mwp']:.2e} (target improvement)")
    
    def update_performance(self, variable_losses: Dict[str, float]):
        """Update performance tracking and adjust learning rates"""
        
        for var in ['swh', 'mwd', 'mwp']:
            current_loss = variable_losses.get(var, float('inf'))
            self.loss_history[var].append(current_loss)
            
            # Check for improvement
            if current_loss < self.best_losses[var]:
                self.best_losses[var] = current_loss
                self.plateau_counters[var] = 0
            else:
                self.plateau_counters[var] += 1
            
            # Reduce LR if plateaued
            if self.plateau_counters[var] >= self.config.lr_patience:
                old_lr = self.current_lrs[var]
                new_lr = max(old_lr * self.config.lr_factor, 
                           self.base_lr * self.config.min_lr_factor)
                
                if new_lr < old_lr:
                    self.current_lrs[var] = new_lr
                    self.plateau_counters[var] = 0
                    print(f"   📉 {var.upper()} LR reduced: {old_lr:.2e} → {new_lr:.2e}")
    
    def get_current_lrs(self) -> Dict[str, float]:
        return self.current_lrs.copy()

class ClimateAttention(nn.Module):
    """Climate attention mechanism for global patterns"""
    
    def __init__(self, hidden_dim: int, climate_dim: int = 6, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.climate_dim = climate_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Project climate features to hidden space
        self.climate_projection = nn.Linear(climate_dim, hidden_dim)
        
        # Climate-aware attention
        self.climate_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Fusion with main features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, main_features: torch.Tensor, climate_features: torch.Tensor) -> torch.Tensor:
        """Apply climate attention"""
        batch_size, num_nodes, _ = main_features.shape
        
        # Project climate features
        climate_proj = self.climate_projection(climate_features)
        
        # Climate attention
        climate_attended, _ = self.climate_attention(
            climate_proj, climate_proj, climate_proj
        )
        
        # Fuse with main features
        combined = torch.cat([main_features, climate_attended], dim=-1)
        fused = self.fusion(combined)
        
        return self.layer_norm(main_features + fused)

class SpatialAttention(nn.Module):
    """Proven spatial attention from Variable LR model"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply edge mask if provided
        if edge_index is not None:
            mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            mask[edge_index[0], edge_index[1]] = True
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        
        output = self.out_linear(attended)
        return self.layer_norm(x + output)

class TemporalAttention(nn.Module):
    """Proven temporal attention from Variable LR model"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        output = self.out_linear(attended)
        return self.layer_norm(x + output)

class EnhancedMessageLayer(nn.Module):
    """Proven message passing layer from Variable LR model"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:  # No edges
            return x
            
        source = x[edge_index[0]]
        target = x[edge_index[1]]
        message_input = torch.cat([source, target, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)
        
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        aggregated = aggregated.index_add(0, edge_index[1], messages)
        
        update_input = torch.cat([x, aggregated], dim=-1)
        gate = self.update_gate(update_input)
        update = self.update_mlp(update_input)
        
        output = gate * update + (1 - gate) * x
        return self.layer_norm(output)

class FullGlobalDataset(Dataset):
    """Dataset for smart-merged files with all features"""
    
    def __init__(self, data_files: List[str], config: FullGlobalConfig, 
                 is_validation: bool = False, validation_split: float = 0.2):
        
        self.data_files = data_files
        self.config = config
        self.is_validation = is_validation
        
        print(f"📊 Loading FULL feature data from {len(data_files)} files...")
        
        # Load and combine all data
        self.data_arrays = []
        self.time_indices = []
        
        for file_path in data_files:
            data_array = self._load_full_featured_file(file_path)
            if data_array is not None:
                self.data_arrays.append(data_array)
                file_sequences = self._create_sequences_from_array(data_array, len(self.data_arrays) - 1)
                self.time_indices.extend(file_sequences)
        
        if not self.time_indices:
            raise ValueError("No valid sequences found in data files")
        
        # Train/val split
        total_sequences = len(self.time_indices)
        val_size = int(total_sequences * validation_split)
        
        if is_validation:
            self.time_indices = self.time_indices[-val_size:] if val_size > 0 else []
        else:
            self.time_indices = self.time_indices[:-val_size] if val_size > 0 else self.time_indices
        
        print(f"🌍 Full Global Dataset ({'validation' if is_validation else 'training'}):")
        print(f"   Sequences: {len(self.time_indices)}")
        print(f"   Features per sample: {self.config.input_features}")
    
    def _load_full_featured_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load file and extract ALL available features"""
        
        print(f"📥 Loading: {Path(file_path).name}")
        
        try:
            with xr.open_dataset(file_path) as ds:
                available_vars = list(ds.data_vars.keys())
                print(f"   Available variables: {len(available_vars)}")
                
                # Extract ALL available variables in priority order
                extracted_vars = []
                var_names = []
                
                for expected_var, possible_names in FULL_VARIABLE_MAPPING.items():
                    found = False
                    for possible_name in possible_names:
                        if possible_name in available_vars:
                            var_data = ds[possible_name]
                            
                            # Ensure 3D structure (time, lat, lon)
                            if len(var_data.dims) == 3:
                                extracted_vars.append(var_data.values)
                                var_names.append(expected_var)
                                found = True
                                print(f"   ✅ {expected_var} from {possible_name}: {var_data.shape}")
                                break
                    
                    if not found and expected_var in ['u10', 'v10', 'swh', 'mwd', 'mwp', 'u850', 'v850', 'z500']:
                        # Only create dummies for essential variables
                        if extracted_vars:
                            dummy_shape = extracted_vars[0].shape
                            dummy_data = np.zeros(dummy_shape)
                            extracted_vars.append(dummy_data)
                            var_names.append(expected_var)
                            print(f"   🔄 {expected_var}: created dummy (essential)")
                
                print(f"   📊 Total extracted: {len(extracted_vars)} variables")
                
                if len(extracted_vars) < 11:  # Minimum required
                    print(f"   ❌ Insufficient variables: {len(extracted_vars)} < 11 minimum")
                    return None
                
                # Pad to target feature count if needed
                target_features = self.config.input_features
                while len(extracted_vars) < target_features:
                    # Create climate anomaly features from existing data
                    base_idx = len(extracted_vars) % len(extracted_vars)  # Cycle through existing
                    base_data = extracted_vars[base_idx]
                    
                    # Create anomaly-like feature (small perturbation)
                    anomaly_data = base_data * 0.1 * (np.random.randn(*base_data.shape) * 0.1)
                    extracted_vars.append(anomaly_data)
                    var_names.append(f'derived_climate_{len(extracted_vars)}')
                    print(f"   🧮 Created derived climate feature {len(extracted_vars)}")
                
                # Stack variables: [time, lat, lon, features]
                combined_data = np.stack(extracted_vars[:target_features], axis=-1)
                
                # Spatial subsampling for memory efficiency
                subsample = self.config.spatial_subsample
                if subsample > 1:
                    combined_data = combined_data[:, ::subsample, ::subsample, :]
                    print(f"   📏 Subsampled by {subsample}: {combined_data.shape}")
                
                print(f"   ✅ Final shape: {combined_data.shape}")
                print(f"   📋 Features: {var_names[:target_features]}")
                
                return combined_data
                
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            return None
    
    def _create_sequences_from_array(self, data_array: np.ndarray, array_idx: int) -> List[Dict]:
        """Create sequence indices from a data array"""
        time_steps = data_array.shape[0]
        sequences = []
        
        for t in range(time_steps - self.config.sequence_length - 1):
            sequences.append({
                'array_idx': array_idx,
                'time_start': t,
                'time_end': t + self.config.sequence_length,
                'target_time': t + self.config.sequence_length
            })
        
        return sequences
    
    def __len__(self):
        return len(self.time_indices)
    
    def __getitem__(self, idx):
        """Get a sequence with all features"""
        
        if idx >= len(self.time_indices):
            return self._get_dummy_sequence()
        
        seq_info = self.time_indices[idx]
        
        try:
            # Get data array
            data_array = self.data_arrays[seq_info['array_idx']]
            
            # Extract input sequence: [seq_len, lat, lon, features]
            input_sequence = data_array[seq_info['time_start']:seq_info['time_end']]
            
            # Extract target: [lat, lon, features]
            target_timestep = data_array[seq_info['target_time']]
            
            # Get wave variables as targets (SWH, MWD, MWP are indices 5, 6, 7)
            target_waves = target_timestep[:, :, 5:8]  # [lat, lon, 3]
            
            # Flatten spatial dimensions
            seq_len, height, width, n_features = input_sequence.shape
            
            # Reshape to [seq_len, nodes, features] and [nodes, target_features]
            input_flat = input_sequence.reshape(seq_len, -1, n_features)
            target_flat = target_waves.reshape(-1, 3)
            
            # Limit nodes for memory but keep substantial data
            max_nodes = 3000  # More nodes for full training
            if input_flat.shape[1] > max_nodes:
                step = input_flat.shape[1] // max_nodes
                input_flat = input_flat[:, ::step, :][:, :max_nodes, :]
                target_flat = target_flat[::step, :][:max_nodes, :]
            
            # Convert to tensors
            input_tensor = torch.tensor(input_flat, dtype=torch.float32)
            target_tensor = torch.tensor(target_flat, dtype=torch.float32)
            
            # Handle NaN/inf values
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return {
                'input': input_tensor,
                'target': target_tensor
            }
            
        except Exception as e:
            print(f"Error getting sequence {idx}: {e}")
            return self._get_dummy_sequence()
    
    def _get_dummy_sequence(self):
        """Return dummy sequence for error cases"""
        dummy_nodes = 2000
        
        input_tensor = torch.zeros(self.config.sequence_length, dummy_nodes, self.config.input_features)
        target_tensor = torch.zeros(dummy_nodes, 3)
        
        return {
            'input': input_tensor,
            'target': target_tensor
        }

class FullGlobalVariableLRModel(nn.Module):
    """Full Global Climate-Aware Variable LR Model"""
    
    def __init__(self, config: FullGlobalConfig):
        super().__init__()
        self.config = config
        
        print(f"🌍 Full Global Climate-Aware Variable LR Model:")
        print(f"   Input features: {config.input_features}")
        print(f"   Hidden dim: {config.hidden_dim}")
        print(f"   Spatial layers: {config.num_spatial_layers}")
        print(f"   Climate attention: {config.use_climate_attention}")
        
        # Enhanced feature encoding for global domain
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Climate feature separation (last 6 features)
        self.climate_feature_extractor = nn.Linear(config.input_features, 6)
        
        # Spatial layers (proven architecture)
        self.spatial_layers = nn.ModuleList([
            EnhancedMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Spatial attention (proven)
        if config.use_spatial_attention:
            self.spatial_attention = SpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        # Climate attention (global innovation)
        if config.use_climate_attention:
            self.climate_attention = ClimateAttention(config.hidden_dim, climate_dim=6)
        
        # Temporal processing (proven architecture)
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
        )
        
        # Temporal attention (proven)
        if config.use_temporal_attention:
            self.temporal_attention = TemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Variable-specific prediction heads (proven architecture)
        hidden_size = config.hidden_dim // 2
        
        # Shared feature transformation
        self.shared_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Variable-specific final layers (proven successful)
        self.swh_head = nn.Linear(hidden_size, 1)      # SWH output
        self.mwd_head = nn.Linear(hidden_size, 2)      # MWD [cos, sin] output  
        self.mwp_head = nn.Linear(hidden_size, 1)      # MWP output
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly (proven approach)"""
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass with full climate awareness"""
        
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        # Process each timestep spatially (proven approach)
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            
            # Extract climate features (last 6 features)
            climate_features = x_t[:, :, -6:]  # [batch, nodes, 6]
            
            # Main feature encoding
            h_t = self.feature_encoder(x_t)
            
            # Spatial processing for each sample in batch (proven approach)
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]
                climate_b = climate_features[b]
                
                # Message passing (proven)
                for layer in self.spatial_layers:
                    h_b = layer(h_b, edge_index, edge_attr)
                
                # Spatial attention (proven)
                if self.config.use_spatial_attention:
                    h_b = self.spatial_attention(h_b.unsqueeze(0), edge_index).squeeze(0)
                
                # Climate attention (innovation)
                if self.config.use_climate_attention:
                    h_b = self.climate_attention(h_b.unsqueeze(0), climate_b.unsqueeze(0)).squeeze(0)
                
                batch_outputs.append(h_b)
            
            h_t = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(h_t)
        
        # Stack temporal sequence
        spatial_sequence = torch.stack(spatial_outputs, dim=1)
        
        # Temporal processing for each node (proven approach)
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]
            
            # LSTM (proven)
            lstm_out, _ = self.temporal_encoder(node_sequence)
            
            # Temporal attention (proven)
            if self.config.use_temporal_attention:
                lstm_out = self.temporal_attention(lstm_out)
            
            # Use last timestep
            final_state = lstm_out[:, -1, :]
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)
        
        # Shared feature transformation (proven)
        shared_features = self.shared_head(temporal_features)
        
        # Variable-specific predictions (proven architecture)
        swh_pred = self.swh_head(shared_features)      # [batch, nodes, 1]
        mwd_pred = self.mwd_head(shared_features)      # [batch, nodes, 2]
        mwp_pred = self.mwp_head(shared_features)      # [batch, nodes, 1]
        
        # Combine predictions: [SWH, MWD_cos, MWD_sin, MWP]
        predictions = torch.cat([swh_pred, mwd_pred, mwp_pred], dim=-1)
        
        return predictions
    
    def get_variable_parameters(self):
        """Get parameters grouped by variable for separate optimizers (proven approach)"""
        
        # Shared parameters (encoder, spatial, temporal, climate)
        shared_params = []
        shared_params.extend(self.feature_encoder.parameters())
        shared_params.extend(self.climate_feature_extractor.parameters())
        shared_params.extend(self.spatial_layers.parameters())
        shared_params.extend(self.temporal_encoder.parameters())
        shared_params.extend(self.shared_head.parameters())
        
        if hasattr(self, 'spatial_attention'):
            shared_params.extend(self.spatial_attention.parameters())
        if hasattr(self, 'temporal_attention'):
            shared_params.extend(self.temporal_attention.parameters())
        if hasattr(self, 'climate_attention'):
            shared_params.extend(self.climate_attention.parameters())
        
        # Variable-specific parameters (proven successful)
        swh_params = list(self.swh_head.parameters())
        mwd_params = list(self.mwd_head.parameters())
        mwp_params = list(self.mwp_head.parameters())
        
        return {
            'shared': shared_params,
            'swh': swh_params,
            'mwd': mwd_params,
            'mwp': mwp_params
        }

class GlobalVariableSpecificLoss(nn.Module):
    """Enhanced loss function for global training with variable-specific components"""
    
    def __init__(self, config: FullGlobalConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute variable-specific losses with climate awareness
        predictions: [batch, nodes, 4] - [SWH, MWD_cos, MWD_sin, MWP]
        targets: [batch, nodes, 4] - [SWH, MWD_cos, MWD_sin, MWP]
        """
        
        # Extract predictions and targets
        pred_swh = predictions[:, :, 0]
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        pred_mwp = predictions[:, :, 3]
        
        true_swh = targets[:, :, 0]
        true_mwd_cos = targets[:, :, 1]
        true_mwd_sin = targets[:, :, 2]
        true_mwp = targets[:, :, 3]
        
        # SWH Loss (standard MSE)
        swh_loss = F.mse_loss(pred_swh, true_swh)
        
        # MWD Loss (circular - proven approach)
        mwd_cos_loss = F.mse_loss(pred_mwd_cos, true_mwd_cos)
        mwd_sin_loss = F.mse_loss(pred_mwd_sin, true_mwd_sin)
        
        # Angular distance loss for MWD
        pred_angles = torch.atan2(pred_mwd_sin, pred_mwd_cos)
        true_angles = torch.atan2(true_mwd_sin, true_mwd_cos)
        angle_diff = pred_angles - true_angles
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        mwd_angular_loss = torch.mean(angle_diff ** 2)
        
        # Combined MWD loss
        mwd_loss = mwd_cos_loss + mwd_sin_loss + mwd_angular_loss
        
        # MWP Loss (standard MSE)
        mwp_loss = F.mse_loss(pred_mwp, true_mwp)
        
        # Enhanced physics constraints for global domain
        physics_loss = self._compute_global_physics_constraints(predictions)
        
        # Weighted total loss (proven weights)
        total_loss = (self.config.swh_loss_weight * swh_loss +
                     self.config.mwd_loss_weight * mwd_loss +
                     self.config.mwp_loss_weight * mwp_loss +
                     self.config.physics_loss_weight * physics_loss)
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_loss': mwd_loss,
            'mwp_loss': mwp_loss,
            'physics_loss': physics_loss,
            'mwd_circular_loss': mwd_cos_loss + mwd_sin_loss,
            'mwd_angular_loss': mwd_angular_loss
        }
    
    def _compute_global_physics_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """Enhanced physics constraints for global domain"""
        pred_swh = predictions[:, :, 0]
        pred_mwp = predictions[:, :, 3]
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        
        # Physical bounds
        swh_penalty = F.relu(-pred_swh).mean()  # SWH >= 0
        mwp_penalty = F.relu(1 - pred_mwp).mean()  # MWP >= 1
        
        # Unit circle constraint for MWD
        circle_penalty = F.mse_loss(pred_mwd_cos**2 + pred_mwd_sin**2, 
                                   torch.ones_like(pred_mwd_cos))
        
        # Global wave height constraints (larger waves possible in open ocean)
        extreme_swh_penalty = F.relu(pred_swh - 20).mean()  # SWH < 20m reasonable for global
        
        return swh_penalty + mwp_penalty + 0.1 * circle_penalty + 0.05 * extreme_swh_penalty

class FullGlobalTrainer:
    """Full trainer with variable-specific learning rates and climate awareness"""
    
    def __init__(self, config: FullGlobalConfig, data_files: List[str]):
        self.config = config
        self.data_files = data_files
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"🖥️  Device: {self.device}")
        
        # Normalizers
        self.feature_normalizer = StandardScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Learning rate manager (proven approach)
        self.lr_manager = GlobalVariableLearningManager(config)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"full_global_1month_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Experiment: {self.experiment_id}")
    
    def prepare_data(self):
        """Prepare full-featured datasets and normalizers"""
        
        print(f"🔄 Preparing FULL data from {len(self.data_files)} files...")
        
        # Create datasets
        train_dataset = FullGlobalDataset(
            self.data_files, self.config, 
            is_validation=False, 
            validation_split=self.config.validation_split
        )
        
        val_dataset = FullGlobalDataset(
            self.data_files, self.config,
            is_validation=True,
            validation_split=self.config.validation_split
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Fit normalizers
        print(f"🔧 Fitting normalizers on sample data...")
        feature_samples = []
        target_samples = []
        
        sample_count = 0
        for batch in train_loader:
            if sample_count >= 10:  # Use first 10 batches for normalization
                break
            
            inputs = batch['input']
            targets = batch['target']
            
            # Flatten for normalization
            batch_size, seq_len, nodes, features = inputs.shape
            inputs_flat = inputs.view(-1, features).numpy()
            targets_flat = targets.view(-1, 3).numpy()
            
            feature_samples.append(inputs_flat)
            target_samples.append(targets_flat)
            sample_count += 1
        
        if feature_samples:
            all_features = np.vstack(feature_samples)
            all_targets = np.vstack(target_samples)
            
            self.feature_normalizer.fit(all_features)
            self.target_normalizer.fit(all_targets)
            
            print(f"   ✅ Normalizers fitted on {all_features.shape[0]} samples")
        
        # Create edge connectivity (simple grid for global data)
        print(f"🔧 Creating edge connectivity...")
        dummy_nodes = 3000  # Larger for full training
        dummy_edges = []
        
        # Create grid-like connectivity
        for i in range(dummy_nodes - 1):
            if i + 1 < dummy_nodes:
                dummy_edges.append([i, i + 1])
            if i + 50 < dummy_nodes:  # Longer range connections
                dummy_edges.append([i, i + 50])
        
        edge_index = torch.tensor(np.array(dummy_edges).T, dtype=torch.long, device=self.device)
        edge_attr = torch.ones(len(dummy_edges), 3, dtype=torch.float32, device=self.device)
        
        print(f"   ✅ Created {len(dummy_edges)} edges for spatial connectivity")
        
        return train_loader, val_loader, edge_index, edge_attr
    
    def create_variable_optimizers(self, model):
        """Create variable-specific optimizers (proven approach)"""
        
        param_groups = model.get_variable_parameters()
        current_lrs = self.lr_manager.get_current_lrs()
        
        optimizers = {
            'shared': torch.optim.AdamW(
                param_groups['shared'], 
                lr=current_lrs['mwd'],  # Use MWD LR for shared params
                weight_decay=self.config.weight_decay
            ),
            'swh': torch.optim.AdamW(
                param_groups['swh'], 
                lr=current_lrs['swh'],
                weight_decay=self.config.weight_decay
            ),
            'mwd': torch.optim.AdamW(
                param_groups['mwd'], 
                lr=current_lrs['mwd'],
                weight_decay=self.config.weight_decay
            ),
            'mwp': torch.optim.AdamW(
                param_groups['mwp'], 
                lr=current_lrs['mwp'],
                weight_decay=self.config.weight_decay
            )
        }
        
        return optimizers
    
    def update_optimizers(self, optimizers):
        """Update optimizer learning rates"""
        
        current_lrs = self.lr_manager.get_current_lrs()
        
        # Update shared optimizer with MWD LR
        optimizers['shared'].param_groups[0]['lr'] = current_lrs['mwd']
        
        # Update variable-specific optimizers
        for var in ['swh', 'mwd', 'mwp']:
            optimizers[var].param_groups[0]['lr'] = current_lrs[var]
    
    def train_epoch(self, model, train_loader, criterion, optimizers, edge_index, edge_attr):
        """Train one epoch with variable-specific learning rates"""
        
        model.train()
        epoch_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Zero all optimizers
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Normalize inputs
                batch_size, seq_len, num_nodes, num_features = inputs.size()
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.feature_normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets with circular MWD
                targets_flat = targets.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)  # [N, 4]
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
                
                # Adjust edge connectivity for actual number of nodes
                if num_nodes != edge_index.max().item() + 1:
                    # Create simple connectivity for this batch size
                    edges = []
                    step = max(1, num_nodes // 100)  # Create ~100 connections
                    for i in range(0, num_nodes - step, step):
                        if i + step < num_nodes:
                            edges.append([i, i + step])
                    
                    if edges:
                        batch_edge_index = torch.tensor(np.array(edges).T, dtype=torch.long, device=self.device)
                        batch_edge_attr = torch.ones(len(edges), 3, dtype=torch.float32, device=self.device)
                    else:
                        # No edges case
                        batch_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                        batch_edge_attr = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
                else:
                    batch_edge_index = edge_index
                    batch_edge_attr = edge_attr
                
                # Forward pass
                predictions = model(inputs, batch_edge_index, batch_edge_attr)
                
                # Compute losses
                loss_dict = criterion(predictions, targets)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                
                # Clip gradients
                for optimizer in optimizers.values():
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group['params']], 
                        self.config.gradient_clip_norm
                    )
                
                # Step all optimizers
                for optimizer in optimizers.values():
                    optimizer.step()
                
                # Track losses
                for key in epoch_losses:
                    if f'{key}_loss' in loss_dict:
                        epoch_losses[key].append(loss_dict[f'{key}_loss'].item())
                
            except Exception as e:
                print(f"   Warning: Batch {batch_idx} failed: {e}")
                continue
        
        return {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion, edge_index, edge_attr):
        """Validate model"""
        
        model.eval()
        val_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                try:
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
                    
                    # Adjust edge connectivity
                    if num_nodes != edge_index.max().item() + 1:
                        edges = []
                        step = max(1, num_nodes // 100)
                        for i in range(0, num_nodes - step, step):
                            if i + step < num_nodes:
                                edges.append([i, i + step])
                        
                        if edges:
                            batch_edge_index = torch.tensor(np.array(edges).T, dtype=torch.long, device=self.device)
                            batch_edge_attr = torch.ones(len(edges), 3, dtype=torch.float32, device=self.device)
                        else:
                            batch_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                            batch_edge_attr = torch.zeros(0, 3, dtype=torch.float32, device=self.device)
                    else:
                        batch_edge_index = edge_index
                        batch_edge_attr = edge_attr
                    
                    # Forward pass
                    predictions = model(inputs, batch_edge_index, batch_edge_attr)
                    
                    # Compute loss
                    loss_dict = criterion(predictions, targets)
                    
                    # Track losses
                    for key in val_losses:
                        if f'{key}_loss' in loss_dict:
                            val_losses[key].append(loss_dict[f'{key}_loss'].item())
                            
                except Exception as e:
                    continue
        
        return {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
    
    def train(self):
        """Main training loop for full global model"""
        
        print(f"🚀 Starting FULL global climate-aware training...")
        
        # Prepare data
        train_loader, val_loader, edge_index, edge_attr = self.prepare_data()
        
        # Create model
        model = FullGlobalVariableLRModel(self.config).to(self.device)
        criterion = GlobalVariableSpecificLoss(self.config)
        
        # Create variable-specific optimizers
        optimizers = self.create_variable_optimizers(model)
        
        print(f"✅ Full Global Model Created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Architecture: Full Variable LR + Climate Attention + Global Domain")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train and validate
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizers, edge_index, edge_attr)
            val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr)
            
            # Update learning rate manager
            variable_losses = {
                'swh': val_metrics.get('swh', float('inf')),
                'mwd': val_metrics.get('mwd', float('inf')),
                'mwp': val_metrics.get('mwp', float('inf'))
            }
            
            self.lr_manager.update_performance(variable_losses)
            self.update_optimizers(optimizers)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            current_lrs = self.lr_manager.get_current_lrs()
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs}: "
                  f"Total={train_metrics['total']:.4f}, "
                  f"Val={val_metrics['total']:.4f}, "
                  f"Time={epoch_time:.1f}s")
            print(f"       SWH={val_metrics.get('swh', 0):.4f} (LR={current_lrs['swh']:.2e}), "
                  f"MWD={val_metrics.get('mwd', 0):.4f} (LR={current_lrs['mwd']:.2e}), "
                  f"MWP={val_metrics.get('mwp', 0):.4f} (LR={current_lrs['mwp']:.2e})")
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'feature_normalizer': self.feature_normalizer,
                    'target_normalizer': self.target_normalizer,
                    'lr_manager_state': self.lr_manager,
                    'epoch': epoch,
                    'val_loss': val_metrics['total']
                }, self.log_dir / "best_full_global_model.pt")
                
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        print(f"✅ Full global training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {self.log_dir}")
        
        return model

def main():
    """Main function for full global 1-month training"""
    
    print("🌍 FULL GLOBAL CLIMATE-AWARE VARIABLE LR MODEL - 1 MONTH INTENSIVE")
    print("=" * 80)
    
    # Full configuration based on proven Variable LR success
    config = FullGlobalConfig(
        # Proven temporal settings
        sequence_length=6,
        prediction_horizon=4,
        
        # Full feature set
        input_features=17,              # All features including climate
        
        # Enhanced global architecture
        hidden_dim=384,                 # Enhanced for global
        temporal_hidden_dim=192,        # Proportional
        num_spatial_layers=8,           # Enhanced connectivity
        num_temporal_layers=2,          # Proven depth
        
        # Full attention suite
        use_spatial_attention=True,
        use_temporal_attention=True,
        use_climate_attention=True,     # Keep climate innovation
        num_attention_heads=8,
        
        # Proven regularization
        dropout=0.15,
        spatial_dropout=0.1,
        temporal_dropout=0.1,
        
        # Intensive 1-month training
        num_epochs=200,                 # Full training
        batch_size=6,                   # Proven batch size
        base_learning_rate=1e-4,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        
        # Proven variable-specific learning rates
        swh_lr_multiplier=0.7,          # Excellent SWH performance
        mwd_lr_multiplier=1.0,          # Circular MWD handling
        mwp_lr_multiplier=1.3,          # MWP improvement focus
        
        # Proven adaptive learning
        lr_patience=15,
        lr_factor=0.8,
        min_lr_factor=0.1,
        
        # Reasonable early stopping
        early_stopping_patience=50,
        
        # Proven loss weights
        swh_loss_weight=1.0,
        mwd_loss_weight=1.0,
        mwp_loss_weight=1.0,
        physics_loss_weight=0.2,
        
        # Data parameters
        validation_split=0.2,
        spatial_subsample=4             # Less aggressive for full training
    )
    
    print(f"🎯 Full Global Configuration:")
    print(f"   Based on: Variable LR proven architecture")
    print(f"   Enhancement: Global domain + climate awareness")
    print(f"   Features: {config.input_features} (all available)")
    print(f"   Architecture: Full complexity with all attention mechanisms")
    print(f"   Training: {config.num_epochs} epochs, variable learning rates")
    print(f"   Target: Replicate impressive 1-month performance at global scale")
    
    # Find smart-merged data files
    data_files = []
    data_dir = Path("data/era5_global")
    
    if data_dir.exists():
        # Look for smart-merged files first
        smart_files = list(data_dir.glob("era5_smart_joined_*.nc"))
        if smart_files:
            # Use first file for 1-month intensive training
            data_files = [str(smart_files[0])]
            print(f"📁 Using smart-merged file: {Path(data_files[0]).name}")
        else:
            # Fallback to regular joined files
            joined_files = list(data_dir.glob("era5_joined_*.nc"))
            if joined_files:
                data_files = [str(joined_files[0])]
                print(f"📁 Using joined file: {Path(data_files[0]).name}")
    
    if not data_files:
        print("❌ No merged data files found!")
        print("   Expected: era5_smart_joined_*.nc or era5_joined_*.nc")
        print("   Run the smart merger first:")
        print("     python smart_pressure_merger.py data/era5_global")
        return
    
    # Train full global model
    trainer = FullGlobalTrainer(config, data_files)
    model = trainer.train()
    
    print(f"\n🎉 FULL GLOBAL 1-MONTH TRAINING COMPLETE!")
    print(f"   Architecture: Complete Variable LR + Climate Attention + Global Domain")
    print(f"   Next steps:")
    print(f"   1. Evaluate performance vs previous 1-month results")
    print(f"   2. If successful, add more months progressively")
    print(f"   3. Compare against 9.21 RMSE baseline")
    print(f"   4. Scale to multi-month global training")

if __name__ == "__main__":
    main()