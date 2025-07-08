#!/usr/bin/env python3
"""
Enhanced Physics-Informed Spatiotemporal Wave Model
Builds on proven circular MWD approach with physics constraints and moderate scaling
SAFE: Preserves working baseline while adding improvements
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import time
import math
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import existing components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import DataConfig, MeshConfig

# Import the proven MWD fixes
from mwd_circular_fixes import VariableSpecificNormalizer, CircularLoss, evaluate_model_with_circular_metrics

@dataclass
class EnhancedPhysicsConfig:
    """Configuration for enhanced physics-informed model"""
    
    # Temporal parameters (proven)
    sequence_length: int = 6           
    prediction_horizon: int = 4        
    
    # SCALED Model architecture
    input_features: int = 11
    hidden_dim: int = 192              # Scaled up from 256
    temporal_hidden_dim: int = 192     # Scaled up from 128
    num_spatial_layers: int = 4        # Scaled up from 6
    num_temporal_layers: int = 2       # Scaled up from 2
    
    # Enhanced attention
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    num_attention_heads: int = 6      # Scaled up from 8
    
    # Regularization (slightly reduced for larger model)
    dropout: float = 0.25              # Reduced from 0.15
    spatial_dropout: float = 0.20      # Reduced from 0.1
    temporal_dropout: float = 0.08     # Reduced from 0.1
    
    # Training parameters
    num_epochs: int = 40               # More epochs for larger model
    batch_size: int = 6                # Reduced for larger model
    learning_rate: float = 8e-5        # Slightly lower for stability
    weight_decay: float = 5e-3
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 12
    
    # ENHANCED Loss weights with physics
    mse_weight: float = 1.0
    circular_weight: float = 1.0
    physics_loss_weight: float = 0.4   # Increased physics influence
    temporal_consistency_weight: float = 0.15
    wave_equation_weight: float = 0.1  # NEW: Wave equation constraints
    energy_conservation_weight: float = 0.05  # NEW: Energy conservation
    
    # Data
    validation_split: float = 0.2
    max_training_sequences: int = 2000  # More sequences for larger model
    
    # Curriculum learning (proven)
    start_with_single_step: bool = True
    single_step_epochs: int = 20

class PhysicsCalculator:
    """Enhanced physics calculations for wave modeling"""
    
    def __init__(self, config: EnhancedPhysicsConfig):
        self.config = config
        self.g = 9.81  # Gravity
        
    def dispersion_relation(self, wave_period: torch.Tensor, water_depth: torch.Tensor) -> torch.Tensor:
        """
        Deep water dispersion relation: ω² = gk
        Returns wavenumber k
        """
        omega = 2 * math.pi / (wave_period + 1e-6)
        k_deep = omega**2 / self.g
        return k_deep
    
    def group_velocity(self, wave_period: torch.Tensor) -> torch.Tensor:
        """
        Group velocity cg = gT/(4π) for deep water
        """
        cg = self.g * wave_period / (4 * math.pi)
        return cg
    
    def wave_energy(self, wave_height: torch.Tensor) -> torch.Tensor:
        """
        Wave energy E = (1/8) * ρ * g * H²
        """
        rho = 1025.0  # Seawater density
        energy = (1/8) * rho * self.g * wave_height**2
        return energy
    
    def wave_steepness(self, wave_height: torch.Tensor, wave_period: torch.Tensor) -> torch.Tensor:
        """
        Wave steepness H/L, where L = gT²/(2π)
        """
        wavelength = self.g * wave_period**2 / (2 * math.pi)
        steepness = wave_height / (wavelength + 1e-6)
        return steepness

# Copy proven classes but with scaling enhancements
class EnhancedSpatialAttention(nn.Module):
    """Enhanced multi-head spatial attention with more heads"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.08)  # Reduced dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Enhanced initialization
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight) 
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
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

class EnhancedTemporalAttention(nn.Module):
    """Enhanced temporal attention with more heads"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.08)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Enhanced initialization
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
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
    """Enhanced message passing with deeper MLPs"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Enhanced message computation with deeper network
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced update mechanism
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
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

class EnhancedPhysicsInformedLoss(nn.Module):
    """Enhanced loss with comprehensive physics constraints"""
    
    def __init__(self, config: EnhancedPhysicsConfig):
        super().__init__()
        self.config = config
        self.physics_calc = PhysicsCalculator(config)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                features: torch.Tensor = None, coordinates: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Enhanced loss with wave physics"""
        
        # Basic circular loss (proven)
        circular_loss_fn = CircularLoss(
            mse_weight=self.config.mse_weight,
            circular_weight=self.config.circular_weight,
            physics_weight=0.1  # Light physics in base circular loss
        )
        
        base_losses = circular_loss_fn(predictions, targets)
        
        # Enhanced physics constraints
        physics_loss = self._compute_wave_physics_loss(predictions)
        energy_loss = self._compute_energy_conservation_loss(predictions)
        steepness_loss = self._compute_wave_steepness_loss(predictions)
        
        # Total enhanced loss
        total_loss = (base_losses['total_loss'] +
                     self.config.wave_equation_weight * physics_loss +
                     self.config.energy_conservation_weight * energy_loss +
                     0.1 * steepness_loss)
        
        return {
            'total_loss': total_loss,
            'base_loss': base_losses['total_loss'],
            'circular_loss': base_losses.get('mwd_circular_loss', torch.tensor(0.0)),
            'angular_loss': base_losses.get('mwd_angular_loss', torch.tensor(0.0)),
            'wave_physics_loss': physics_loss,
            'energy_loss': energy_loss,
            'steepness_loss': steepness_loss
        }
    
    def _compute_wave_physics_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Enhanced wave physics constraints"""
        
        # Extract circular predictions [SWH, MWD_cos, MWD_sin, MWP]
        swh = predictions[:, :, 0]      # [batch, nodes]
        mwd_cos = predictions[:, :, 1]
        mwd_sin = predictions[:, :, 2]
        mwp = predictions[:, :, 3]      # [batch, nodes]
        
        # Physical bounds
        swh_penalty = F.relu(-swh).mean() + F.relu(swh - 20).mean()  # 0 <= SWH <= 20
        mwp_penalty = F.relu(1 - mwp).mean() + F.relu(mwp - 25).mean()  # 1 <= MWP <= 25
        
        # Unit circle constraint for MWD
        circle_penalty = F.mse_loss(mwd_cos**2 + mwd_sin**2, torch.ones_like(mwd_cos))
        
        # Wave steepness constraint (H/L < 0.14 for stability)
        steepness = self.physics_calc.wave_steepness(swh, mwp)
        steepness_penalty = F.relu(steepness - 0.14).mean()
        
        return swh_penalty + mwp_penalty + circle_penalty + steepness_penalty
    
    def _compute_energy_conservation_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Energy conservation constraints"""
        
        swh = predictions[:, :, 0]
        energy = self.physics_calc.wave_energy(swh)
        
        # Energy should be smooth across neighboring nodes (simplified)
        energy_variance_penalty = torch.var(energy, dim=1).mean()
        
        return energy_variance_penalty
    
    def _compute_wave_steepness_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Wave steepness physical constraints"""
        
        swh = predictions[:, :, 0]
        mwp = predictions[:, :, 3]
        
        steepness = self.physics_calc.wave_steepness(swh, mwp)
        
        # Steepness should be reasonable (0.01 < H/L < 0.14)
        steepness_bounds = F.relu(0.01 - steepness).mean() + F.relu(steepness - 0.14).mean()
        
        return steepness_bounds

class EnhancedSpatioTemporalWaveGNN(nn.Module):
    """Enhanced spatiotemporal model with physics and scaling"""
    
    def __init__(self, config: EnhancedPhysicsConfig):
        super().__init__()
        self.config = config
        
        # Enhanced feature encoding with deeper network
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Enhanced spatial layers (more layers)
        self.spatial_layers = nn.ModuleList([
            EnhancedMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Enhanced spatial attention
        if config.use_spatial_attention:
            self.spatial_attention = EnhancedSpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        # Enhanced temporal processing (3-layer LSTM)
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
        )
        
        # Enhanced temporal attention
        if config.use_temporal_attention:
            self.temporal_attention = EnhancedTemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Enhanced prediction heads with deeper networks
        self.single_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout // 2),
            nn.Linear(config.hidden_dim // 2, 4)  # [SWH, MWD_cos, MWD_sin, MWP]
        )
        
        self.multi_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 4 * config.prediction_horizon)
        )
        
        # Enhanced physics constraints layer
        self.physics_layer = nn.Sequential(
            nn.Linear(4 * config.prediction_horizon, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 4 * config.prediction_horizon)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Enhanced weight initialization"""
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
                multi_step: bool = False) -> torch.Tensor:
        """Enhanced forward pass with physics integration"""
        
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        # Enhanced spatial processing
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            h_t = self.feature_encoder(x_t)
            
            # Enhanced spatial processing for each sample in batch
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]
                
                # Enhanced message passing with residual connections
                for i, layer in enumerate(self.spatial_layers):
                    h_new = layer(h_b, edge_index, edge_attr)
                    # Add residual connection every 2 layers
                    if i % 2 == 1:
                        h_b = h_new + h_b
                    else:
                        h_b = h_new
                
                # Enhanced spatial attention
                if self.config.use_spatial_attention:
                    h_b = self.spatial_attention(h_b.unsqueeze(0), edge_index).squeeze(0)
                
                batch_outputs.append(h_b)
            
            h_t = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(h_t)
        
        # Enhanced temporal processing
        spatial_sequence = torch.stack(spatial_outputs, dim=1)
        
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]
            
            # Enhanced LSTM
            lstm_out, _ = self.temporal_encoder(node_sequence)
            
            # Enhanced temporal attention
            if self.config.use_temporal_attention:
                lstm_out = self.temporal_attention(lstm_out)
            
            final_state = lstm_out[:, -1, :]
            temporal_outputs.append(final_state)
        
        temporal_features = torch.stack(temporal_outputs, dim=1)
        
        # Enhanced predictions with physics
        if multi_step:
            predictions = self.multi_step_head(temporal_features)
            # Enhanced physics constraints
            physics_adjustment = self.physics_layer(predictions)
            predictions = predictions + 0.15 * physics_adjustment  # Stronger physics influence
        else:
            predictions = self.single_step_head(temporal_features)
        
        return predictions

# Copy proven dataset class
class SpatioTemporalDataset(Dataset):
    """Dataset for spatiotemporal sequences (proven implementation)"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: EnhancedPhysicsConfig, 
                 start_time: int = 0, end_time: int = 500):  # More data for enhanced model
        self.mesh_loader = mesh_loader
        self.config = config
        self.sequences = []
        
        print(f"Building enhanced spatiotemporal dataset from timesteps {start_time} to {end_time}...")
        
        for t in range(start_time, end_time - config.sequence_length - config.prediction_horizon + 1):
            try:
                # Input sequence
                input_features = []
                for i in range(config.sequence_length):
                    features_data = mesh_loader.load_features(time_idx=t + i)
                    features = torch.tensor(features_data['features'], dtype=torch.float32)
                    features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
                    input_features.append(features)
                
                # Target sequence
                target_features = []
                for i in range(config.prediction_horizon):
                    features_data = mesh_loader.load_features(time_idx=t + config.sequence_length + i)
                    targets = torch.tensor(features_data['features'][:, [3, 4, 5]], dtype=torch.float32)
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
                    target_features.append(targets)
                
                # Stack sequences
                input_tensor = torch.stack(input_features, dim=0)
                target_tensor = torch.stack(target_features, dim=0)
                
                self.sequences.append({
                    'input': input_tensor,
                    'target': target_tensor,
                    'single_step_target': target_tensor[0]
                })
                
                if len(self.sequences) >= config.max_training_sequences:
                    break
                    
            except Exception as e:
                continue
        
        print(f"Created {len(self.sequences)} enhanced spatiotemporal sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class EnhancedSpatioTemporalTrainer:
    """Enhanced trainer with physics constraints and scaling"""
    
    def __init__(self, config: EnhancedPhysicsConfig):
        self.config = config
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Proven normalizers
        self.normalizer = StandardScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"enhanced_physics_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Enhanced Physics-Informed Training: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup data with multi-year loading"""
        
        print("📊 Setting up enhanced multi-year dataset...")
        
        # Load data components
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Multi-year data loading (2019-2021 proven)
        print("🔄 Loading multi-year ERA5 data...")
        all_atmo_data = []
        all_wave_data = []
        
        for year in [2019, 2020, 2021]:
            for month in range(1, 13):
                try:
                    era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
                    all_atmo_data.append(era5_atmo)
                    all_wave_data.append(era5_waves)
                    print(f"   ✅ Loaded {year}-{month:02d}")
                except Exception as e:
                    print(f"   ⚠️  Skipping {year}-{month:02d}: {e}")
                    continue
        
        if not all_atmo_data:
            raise ValueError("No data loaded!")
        
        # Concatenate all data
        import xarray as xr
        era5_atmo = xr.concat(all_atmo_data, dim='valid_time')
        era5_waves = xr.concat(all_wave_data, dim='valid_time')
        
        print(f"   ✅ Combined dataset: {len(era5_atmo.valid_time)} timesteps")
        
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh with caching
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create enhanced dataset
        dataset = SpatioTemporalDataset(mesh_loader, self.config, start_time=0, end_time=len(era5_atmo.valid_time)-10)
        
        # Fit normalizers (proven approach)
        print("🔧 Fitting enhanced normalizers...")
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(200, len(dataset)), 20):
            sample = dataset[i]
            features = sample['input'].numpy()
            targets = sample['single_step_target'].numpy()
            
            features_flat = features.reshape(-1, features.shape[-1])
            sample_features.append(features_flat)
            sample_targets.append(targets)
        
        if sample_features:
            all_features = np.vstack(sample_features)
            all_targets = np.vstack(sample_targets)
            
            self.normalizer.fit(all_features)
            self.target_normalizer.fit(all_targets)
        
        # Split dataset
        val_size = int(self.config.validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        
        # Setup graph connectivity
        region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)
        
        print(f"✅ Enhanced data setup complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Nodes: {len(region_indices)}")
        print(f"   Edges: {edge_index.shape[1]}")
        
        return train_loader, val_loader, edge_index, edge_attr
    
    def train_epoch(self, model, train_loader, criterion, optimizer, edge_index, edge_attr, single_step_mode: bool = True):
        """Enhanced training epoch with physics loss tracking"""
        
        model.train()
        epoch_losses = {
            'total': [], 'base': [], 'circular': [], 'angular': [], 
            'wave_physics': [], 'energy': [], 'steepness': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = batch['input'].to(self.device)
            
            if single_step_mode:
                targets = batch['single_step_target'].to(self.device)
            else:
                targets = batch['target'].to(self.device)
                targets = targets.transpose(1, 2).contiguous()
                targets = targets.view(targets.size(0), targets.size(1), -1)
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            inputs_norm = self.normalizer.transform(inputs_flat)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets with circular MWD
            if single_step_mode:
                targets_flat = targets.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
            else:
                batch_size, num_nodes, total_features = targets.size()
                horizon = total_features // 3
                targets_reshaped = targets.view(batch_size, num_nodes, horizon, 3)
                targets_flat = targets_reshaped.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, horizon * 4)
            
            # Forward pass with enhanced model
            predictions = model(inputs, edge_index, edge_attr, multi_step=not single_step_mode)
            
            # Enhanced physics loss
            loss_dict = criterion(predictions, targets, features=inputs)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            
            # Track enhanced losses
            for key in epoch_losses:
                loss_key = f'{key}_loss'
                if loss_key in loss_dict:
                    epoch_losses[key].append(loss_dict[loss_key].item())
        
        return {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion, edge_index, edge_attr, single_step_mode: bool = True):
        """Enhanced validation with physics loss tracking"""
        
        model.eval()
        val_losses = {
            'total': [], 'base': [], 'circular': [], 'angular': [], 
            'wave_physics': [], 'energy': [], 'steepness': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                
                if single_step_mode:
                    targets = batch['single_step_target'].to(self.device)
                else:
                    targets = batch['target'].to(self.device)
                    targets = targets.transpose(1, 2).contiguous()
                    targets = targets.view(targets.size(0), targets.size(1), -1)
                
                # Normalize inputs
                batch_size, seq_len, num_nodes, num_features = inputs.size()
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets
                if single_step_mode:
                    targets_flat = targets.view(-1, 3).cpu().numpy()
                    targets_norm = self.target_normalizer.transform_targets(targets_flat)
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, 4)
                else:
                    batch_size, num_nodes, total_features = targets.size()
                    horizon = total_features // 3
                    targets_reshaped = targets.view(batch_size, num_nodes, horizon, 3)
                    targets_flat = targets_reshaped.view(-1, 3).cpu().numpy()
                    targets_norm = self.target_normalizer.transform_targets(targets_flat)
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, horizon * 4)
                
                # Forward pass
                predictions = model(inputs, edge_index, edge_attr, multi_step=not single_step_mode)
                
                # Compute loss
                loss_dict = criterion(predictions, targets, features=inputs)
                
                # Track losses
                for key in val_losses:
                    loss_key = f'{key}_loss'
                    if loss_key in loss_dict:
                        val_losses[key].append(loss_dict[loss_key].item())
        
        return {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
    
    def train(self):
        """Enhanced training loop with physics tracking"""
        
        print(f"🚀 Starting enhanced physics-informed training...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        
        # Create enhanced model
        model = EnhancedSpatioTemporalWaveGNN(self.config).to(self.device)
        criterion = EnhancedPhysicsInformedLoss(self.config)
        
        # Enhanced optimizer with lower LR for stability
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        print(f"✅ Enhanced model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Hidden dim: {self.config.hidden_dim} (vs 256 baseline)")
        print(f"   Spatial layers: {self.config.num_spatial_layers} (vs 6 baseline)")
        print(f"   Temporal layers: {self.config.num_temporal_layers} (vs 2 baseline)")
        print(f"   Attention heads: {self.config.num_attention_heads} (vs 8 baseline)")
        print(f"   Physics loss weights: Wave={self.config.wave_equation_weight}, Energy={self.config.energy_conservation_weight}")
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'train_physics': [], 'val_physics': [],
            'train_energy': [], 'val_energy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Enhanced single-step training
        print(f"\n📈 Enhanced single-step training ({self.config.single_step_epochs} epochs)")
        
        for epoch in range(self.config.single_step_epochs):
            start_time = time.time()
            
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, edge_index, edge_attr, single_step_mode=True)
            val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr, single_step_mode=True)
            
            scheduler.step()
            
            # Track enhanced history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['train_physics'].append(train_metrics.get('wave_physics', 0))
            history['val_physics'].append(val_metrics.get('wave_physics', 0))
            history['train_energy'].append(train_metrics.get('energy', 0))
            history['val_energy'].append(val_metrics.get('energy', 0))
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{self.config.single_step_epochs}: "
                  f"Total={train_metrics['total']:.4f}, "
                  f"Val={val_metrics['total']:.4f}, "
                  f"Physics={train_metrics.get('wave_physics', 0):.4f}, "
                  f"Energy={train_metrics.get('energy', 0):.4f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Time={epoch_time:.1f}s")
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'feature_normalizer': self.normalizer,
                    'target_normalizer': self.target_normalizer,
                    'edge_index': edge_index.cpu(),
                    'edge_attr': edge_attr.cpu(),
                    'epoch': epoch,
                    'val_loss': val_metrics['total']
                }, self.log_dir / "best_enhanced_model.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        model_data = {
            'experiment_id': self.experiment_id,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'feature_normalizer': self.normalizer,
            'target_normalizer': self.target_normalizer,
            'edge_index': edge_index.cpu(),
            'edge_attr': edge_attr.cpu(),
            'training_history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.log_dir / "enhanced_physics_model.pt"
        torch.save(model_data, model_path)
        
        print(f"✅ Enhanced training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Enhanced model saved: {model_path}")
        print(f"📊 Expected improvement over 13.0 RMSE baseline")
        
        return model, history, edge_index, edge_attr

def main():
    """Main function to train enhanced physics-informed model"""
    
    print("🌊 ENHANCED PHYSICS-INFORMED SPATIOTEMPORAL WAVE MODEL")
    print("=" * 70)
    
    # Enhanced configuration
    config = EnhancedPhysicsConfig(
        # Scaled architecture
        hidden_dim=192,
        temporal_hidden_dim=192,
        num_spatial_layers=4,
        num_temporal_layers=2,
        num_attention_heads=6,
        
        # Enhanced physics
        wave_equation_weight=0.05,
        energy_conservation_weight=0.01,
        physics_loss_weight=0.4,
        
        # Training
        # num_epochs=40,
        num_epochs=10,
        batch_size=6,
        learning_rate=8e-5,
        single_step_epochs=25,
        
        # Data
        max_training_sequences=2000
    )
    
    print(f"🔧 Enhanced Configuration:")
    print(f"   Model scaling: 256→{config.hidden_dim} hidden, 6→{config.num_spatial_layers} spatial, 2→{config.num_temporal_layers} temporal")
    print(f"   Physics constraints: Wave equation + Energy conservation + Steepness")
    print(f"   Expected parameters: ~8-12M (vs 3.9M baseline)")
    print(f"   Target: <10 RMSE (vs 13.0 baseline)")
    
    # Train enhanced model
    trainer = EnhancedSpatioTemporalTrainer(config)
    model, history, edge_index, edge_attr = trainer.train()
    
    print(f"\n🎉 Enhanced physics-informed training complete!")
    print(f"   Ready for evaluation against 13.0 RMSE baseline")

if __name__ == "__main__":
    main()