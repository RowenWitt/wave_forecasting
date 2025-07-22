#!/usr/bin/env python3
"""
Complete Spatiotemporal Wave Forecasting Model
Single-file implementation with training and evaluation
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

@dataclass
class SpatioTemporalConfig:
    """Configuration for spatiotemporal wave model"""
    
    # Temporal parameters
    sequence_length: int = 6           # Input sequence length (t-5 to t0)
    prediction_horizon: int = 4        # Predict t+1 to t+4
    
    # Model architecture
    input_features: int = 11
    hidden_dim: int = 256
    temporal_hidden_dim: int = 128
    num_spatial_layers: int = 6        # Reasonable number of spatial layers
    num_temporal_layers: int = 2       # LSTM layers
    
    # Attention
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    num_attention_heads: int = 8
    
    # Regularization (moderate)
    dropout: float = 0.15
    spatial_dropout: float = 0.1
    temporal_dropout: float = 0.1
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 8               # Smaller batches for sequences
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Loss weights
    physics_loss_weight: float = 0.2
    temporal_consistency_weight: float = 0.1
    
    # Data
    validation_split: float = 0.2
    max_training_sequences: int = 1000  # Reasonable limit
    
    # Curriculum learning
    start_with_single_step: bool = True
    single_step_epochs: int = 15

class SpatialAttention(nn.Module):
    """Multi-head spatial attention for graph nodes"""
    
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
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        """
        batch_size, num_nodes, hidden_dim = x.size()
        
        # Compute Q, K, V
        Q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create edge mask
        if edge_index is not None:
            mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            mask[edge_index[0], edge_index[1]] = True
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
            
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        
        output = self.out_linear(attended)
        return self.layer_norm(x + output)

class TemporalAttention(nn.Module):
    """Multi-head temporal attention"""
    
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
        """
        x: [batch_size, seq_len, hidden_dim]
        """
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
    """Enhanced message passing layer"""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update gate
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
        """
        x: [num_nodes, hidden_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        # Compute messages
        source = x[edge_index[0]]
        target = x[edge_index[1]]
        message_input = torch.cat([source, target, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)
        
        # Aggregate messages
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        aggregated = aggregated.index_add(0, edge_index[1], messages)
        
        # Update with gating
        update_input = torch.cat([x, aggregated], dim=-1)
        gate = self.update_gate(update_input)
        update = self.update_mlp(update_input)
        
        output = gate * update + (1 - gate) * x
        return self.layer_norm(output)

class SpatioTemporalWaveGNN(nn.Module):
    """Complete spatiotemporal wave prediction model"""
    
    def __init__(self, config: SpatioTemporalConfig):
        super().__init__()
        self.config = config
        
        # Feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Spatial layers
        self.spatial_layers = nn.ModuleList([
            EnhancedMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Spatial attention
        if config.use_spatial_attention:
            self.spatial_attention = SpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
        )
        
        # Temporal attention
        if config.use_temporal_attention:
            self.temporal_attention = TemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Prediction heads
        self.single_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # [SWH, MWD, MWP]
        )
        
        self.multi_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 3 * config.prediction_horizon)
        )
        
        # Physics constraints
        self.physics_layer = nn.Linear(3 * config.prediction_horizon, 3 * config.prediction_horizon)
        
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
                multi_step: bool = False) -> torch.Tensor:
        """
        x: [batch_size, seq_len, num_nodes, input_features]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_features]
        """
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        # Process each timestep spatially
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch_size, num_nodes, input_features]
            
            # Encode features
            h_t = self.feature_encoder(x_t)  # [batch_size, num_nodes, hidden_dim]
            
            # Spatial processing for each sample in batch
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]  # [num_nodes, hidden_dim]
                
                # Message passing
                for layer in self.spatial_layers:
                    h_b = layer(h_b, edge_index, edge_attr)
                
                # Spatial attention
                if self.config.use_spatial_attention:
                    h_b = self.spatial_attention(h_b.unsqueeze(0), edge_index).squeeze(0)
                
                batch_outputs.append(h_b)
            
            h_t = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(h_t)
        
        # Stack temporal sequence
        spatial_sequence = torch.stack(spatial_outputs, dim=1)  # [batch_size, seq_len, num_nodes, hidden_dim]
        
        # Temporal processing for each node
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]  # [batch_size, seq_len, hidden_dim]
            
            # LSTM
            lstm_out, _ = self.temporal_encoder(node_sequence)
            
            # Temporal attention
            if self.config.use_temporal_attention:
                lstm_out = self.temporal_attention(lstm_out)
            
            # Use last timestep
            final_state = lstm_out[:, -1, :]  # [batch_size, temporal_hidden_dim]
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)  # [batch_size, num_nodes, temporal_hidden_dim]
        
        # Generate predictions
        if multi_step:
            predictions = self.multi_step_head(temporal_features)
            # Apply physics constraints
            physics_adjustment = self.physics_layer(predictions)
            predictions = predictions + 0.1 * physics_adjustment  # Light physics constraint
        else:
            predictions = self.single_step_head(temporal_features)
        
        return predictions

class SpatioTemporalDataset(Dataset):
    """Dataset for spatiotemporal sequences"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: SpatioTemporalConfig, 
                 start_time: int = 0, end_time: int = 100):
        self.mesh_loader = mesh_loader
        self.config = config
        self.sequences = []
        
        print(f"Building spatiotemporal dataset from timesteps {start_time} to {end_time}...")
        
        # Build sequences
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
                    targets = torch.tensor(features_data['features'][:, [3, 4, 5]], dtype=torch.float32)  # SWH, MWD, MWP
                    targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
                    target_features.append(targets)
                
                # Stack sequences
                input_tensor = torch.stack(input_features, dim=0)  # [seq_len, num_nodes, features]
                target_tensor = torch.stack(target_features, dim=0)  # [horizon, num_nodes, 3]
                
                self.sequences.append({
                    'input': input_tensor,
                    'target': target_tensor,
                    'single_step_target': target_tensor[0]  # First timestep for single-step training
                })
                
                if len(self.sequences) >= config.max_training_sequences:
                    break
                    
            except Exception as e:
                print(f"Skipping timestep {t}: {e}")
                continue
        
        print(f"Created {len(self.sequences)} spatiotemporal sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class SpatioTemporalLoss(nn.Module):
    """Loss function for spatiotemporal model"""
    
    def __init__(self, config: SpatioTemporalConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                multi_step: bool = False) -> Dict[str, torch.Tensor]:
        """
        predictions: Model predictions
        targets: Ground truth targets
        multi_step: Whether this is multi-step prediction
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics constraints
        physics_loss = self._compute_physics_loss(predictions, multi_step)
        
        # Temporal consistency (for multi-step)
        temporal_loss = torch.tensor(0.0, device=predictions.device)
        if multi_step and predictions.size(-1) > 3:
            temporal_loss = self._compute_temporal_consistency(predictions)
        
        # Total loss
        total_loss = (mse_loss + 
                     self.config.physics_loss_weight * physics_loss +
                     self.config.temporal_consistency_weight * temporal_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss,
            'temporal_loss': temporal_loss
        }
    
    def _compute_physics_loss(self, predictions: torch.Tensor, multi_step: bool) -> torch.Tensor:
        """Compute physics-based constraints"""
        if multi_step:
            # Reshape multi-step predictions
            batch_size, num_nodes, pred_dim = predictions.size()
            horizon = pred_dim // 3
            pred_reshaped = predictions.view(batch_size, num_nodes, horizon, 3)
            
            swh = pred_reshaped[:, :, :, 0]
            mwd = pred_reshaped[:, :, :, 1]
            mwp = pred_reshaped[:, :, :, 2]
        else:
            swh = predictions[:, :, 0]
            mwd = predictions[:, :, 1]
            mwp = predictions[:, :, 2]
        
        # Physics constraints
        swh_penalty = F.relu(-swh).mean()  # SWH >= 0
        mwd_penalty = F.relu(mwd - 360).mean() + F.relu(-mwd).mean()  # 0 <= MWD <= 360
        mwp_penalty = F.relu(mwp - 25).mean() + F.relu(1 - mwp).mean()  # 1 <= MWP <= 25
        
        return swh_penalty + mwd_penalty + mwp_penalty
    
    def _compute_temporal_consistency(self, predictions: torch.Tensor) -> torch.Tensor:
        """Encourage temporal smoothness in multi-step predictions"""
        batch_size, num_nodes, pred_dim = predictions.size()
        horizon = pred_dim // 3
        pred_reshaped = predictions.view(batch_size, num_nodes, horizon, 3)
        
        if horizon > 1:
            temporal_diff = pred_reshaped[:, :, 1:, :] - pred_reshaped[:, :, :-1, :]
            return torch.mean(temporal_diff ** 2)
        
        return torch.tensor(0.0, device=predictions.device)

class SimpleNormalizer:
    """Simple feature normalizer"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit scalers"""
        valid_features = features[~np.isnan(features).any(axis=1)]
        valid_targets = targets[~np.isnan(targets).any(axis=1)]
        
        self.feature_scaler.fit(valid_features)
        self.target_scaler.fit(valid_targets)
        self.fitted = True
        
        print(f"Normalizer fitted on {len(valid_features)} feature samples and {len(valid_targets)} target samples")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(features)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(targets)
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(targets)

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class SpatioTemporalTrainer:
    """Complete trainer for spatiotemporal wave model"""
    
    def __init__(self, config: SpatioTemporalConfig):
        self.config = config
        # Device setup with MPS support for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.normalizer = SimpleNormalizer()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"spatiotemporal_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Spatiotemporal Training Experiment: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup data loaders"""
        
        print("📊 Setting up spatiotemporal dataset...")
        
        # Load data components
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load 2020 data
        era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create dataset
        dataset = SpatioTemporalDataset(mesh_loader, self.config, start_time=0, end_time=100)
        
        # Fit normalizer
        print("🔧 Fitting normalizer...")
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(100, len(dataset)), 10):
            sample = dataset[i]
            features = sample['input'].numpy()  # [seq_len, num_nodes, features]
            targets = sample['single_step_target'].numpy()  # [num_nodes, 3]
            
            # Flatten features for normalization
            features_flat = features.reshape(-1, features.shape[-1])
            
            sample_features.append(features_flat)
            sample_targets.append(targets)
        
        if sample_features:
            all_features = np.vstack(sample_features)
            all_targets = np.vstack(sample_targets)
            self.normalizer.fit(all_features, all_targets)
        
        # Split dataset
        val_size = int(self.config.validation_split * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type in ['cuda', 'mps'] else False
        )
        
        # Setup graph connectivity
        region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)
        
        print(f"✅ Data setup complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Nodes: {len(region_indices)}")
        print(f"   Edges: {edge_index.shape[1]}")
        
        return train_loader, val_loader, edge_index, edge_attr
    
    def train_epoch(self, model, train_loader, criterion, optimizer, edge_index, edge_attr, 
                   single_step_mode: bool = True):
        """Train one epoch"""
        
        model.train()
        epoch_losses = {'total': [], 'mse': [], 'physics': [], 'temporal': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get batch data
            inputs = batch['input'].to(self.device)  # [batch_size, seq_len, num_nodes, features]
            
            if single_step_mode:
                targets = batch['single_step_target'].to(self.device)  # [batch_size, num_nodes, 3]
            else:
                targets = batch['target'].to(self.device)  # [batch_size, horizon, num_nodes, 3]
                # Reshape for multi-step prediction
                targets = targets.transpose(1, 2).contiguous()  # [batch_size, num_nodes, horizon, 3]
                targets = targets.view(targets.size(0), targets.size(1), -1)  # [batch_size, num_nodes, horizon*3]
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            inputs_norm = self.normalizer.transform_features(inputs_flat)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets
            if single_step_mode:
                # Single step: [batch_size, num_nodes, 3] -> normalize as-is
                targets_flat = targets.view(-1, targets.size(-1)).cpu().numpy()
                targets_norm = self.normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, -1)
            else:
                # Multi step: [batch_size, num_nodes, 12] -> reshape to [batch_size*num_nodes*4, 3] -> normalize -> reshape back
                batch_size, num_nodes, total_features = targets.size()
                horizon = total_features // 3  # Should be 4
                
                # Reshape to [batch_size, num_nodes, horizon, 3]
                targets_reshaped = targets.view(batch_size, num_nodes, horizon, 3)
                # Flatten to [batch_size*num_nodes*horizon, 3]
                targets_flat = targets_reshaped.view(-1, 3).cpu().numpy()
                # Normalize
                targets_norm = self.normalizer.transform_targets(targets_flat)
                # Reshape back to [batch_size, num_nodes, horizon*3]
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, horizon * 3)
            
            # Forward pass
            predictions = model(inputs, edge_index, edge_attr, multi_step=not single_step_mode)
            
            # Compute loss
            loss_dict = criterion(predictions, targets, multi_step=not single_step_mode)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            
            optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                if f'{key}_loss' in loss_dict:
                    epoch_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion, edge_index, edge_attr, single_step_mode: bool = True):
        """Validate model"""
        
        model.eval()
        val_losses = {'total': [], 'mse': [], 'physics': [], 'temporal': []}
        
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
                inputs_norm = self.normalizer.transform_features(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets
                if single_step_mode:
                    # Single step: [batch_size, num_nodes, 3] -> normalize as-is
                    targets_flat = targets.view(-1, targets.size(-1)).cpu().numpy()
                    targets_norm = self.normalizer.transform_targets(targets_flat)
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, -1)
                else:
                    # Multi step: [batch_size, num_nodes, 12] -> reshape to [batch_size*num_nodes*4, 3] -> normalize -> reshape back
                    batch_size, num_nodes, total_features = targets.size()
                    horizon = total_features // 3  # Should be 4
                    
                    # Reshape to [batch_size, num_nodes, horizon, 3]
                    targets_reshaped = targets.view(batch_size, num_nodes, horizon, 3)
                    # Flatten to [batch_size*num_nodes*horizon, 3]
                    targets_flat = targets_reshaped.view(-1, 3).cpu().numpy()
                    # Normalize
                    targets_norm = self.normalizer.transform_targets(targets_flat)
                    # Reshape back to [batch_size, num_nodes, horizon*3]
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, horizon * 3)
                
                # Forward pass
                predictions = model(inputs, edge_index, edge_attr, multi_step=not single_step_mode)
                
                # Compute loss
                loss_dict = criterion(predictions, targets, multi_step=not single_step_mode)
                
                # Track losses
                for key in val_losses:
                    if f'{key}_loss' in loss_dict:
                        val_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in val_losses.items()}
    
    def test_feature_sensitivity(self, model, val_loader, edge_index, edge_attr):
        """Test feature sensitivity"""
        
        model.eval()
        
        # Get a sample batch
        batch = next(iter(val_loader))
        inputs = batch['input'].to(self.device)
        
        # Normalize inputs
        batch_size, seq_len, num_nodes, num_features = inputs.size()
        inputs_flat = inputs.view(-1, num_features).cpu().numpy()
        inputs_norm = self.normalizer.transform_features(inputs_flat)
        inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
        inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
        
        # Use first sample only
        sample_input = inputs[0:1]  # [1, seq_len, num_nodes, features]
        
        with torch.no_grad():
            # Baseline prediction
            baseline_pred = model(sample_input, edge_index, edge_attr, multi_step=False)
            
            # Test sensitivity to each feature
            sensitivities = []
            for feat_idx in range(num_features):
                modified_input = sample_input.clone()
                modified_input[:, :, :, feat_idx] += 1.0  # Add 1 standard deviation
                
                modified_pred = model(modified_input, edge_index, edge_attr, multi_step=False)
                sensitivity = torch.mean(torch.abs(modified_pred - baseline_pred))
                sensitivities.append(sensitivity.item())
            
            avg_sensitivity = np.mean(sensitivities)
            max_sensitivity = np.max(sensitivities)
            
            return avg_sensitivity, max_sensitivity, sensitivities
    
    def train(self):
        """Main training loop"""
        
        print(f"🚀 Starting spatiotemporal training...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        
        # Create model
        model = SpatioTemporalWaveGNN(self.config).to(self.device)
        criterion = SpatioTemporalLoss(self.config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        print(f"✅ Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {next(model.parameters()).device}")
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'feature_sensitivity': []
        }
        
        # Phase 1: Single-step training
        if self.config.start_with_single_step:
            print(f"\n📈 Phase 1: Single-step training ({self.config.single_step_epochs} epochs)")
            
            for epoch in range(self.config.single_step_epochs):
                start_time = time.time()
                
                # Train and validate
                train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, 
                                               edge_index, edge_attr, single_step_mode=True)
                val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr, 
                                          single_step_mode=True)
                
                # Feature sensitivity
                avg_sens, max_sens, _ = self.test_feature_sensitivity(model, val_loader, edge_index, edge_attr)
                
                # Update scheduler
                scheduler.step()
                
                # Track history
                history['train_loss'].append(train_metrics['total'])
                history['val_loss'].append(val_metrics['total'])
                history['train_mse'].append(train_metrics['mse'])
                history['val_mse'].append(val_metrics['mse'])
                history['feature_sensitivity'].append(avg_sens)
                
                # Progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1:2d}/{self.config.single_step_epochs}: "
                      f"Train={train_metrics['total']:.4f}, "
                      f"Val={val_metrics['total']:.4f}, "
                      f"Sens={avg_sens:.4f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time={epoch_time:.1f}s")
                
                # Early stopping
                if early_stopping(val_metrics['total'], model):
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    model.load_state_dict(early_stopping.best_model_state)
                    break
        
        # Phase 2: Multi-step training
        remaining_epochs = self.config.num_epochs - self.config.single_step_epochs
        if remaining_epochs > 0:
            print(f"\n📈 Phase 2: Multi-step training ({remaining_epochs} epochs)")
            
            # Reset early stopping for phase 2
            early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
            
            for epoch in range(remaining_epochs):
                start_time = time.time()
                
                # Train and validate
                train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, 
                                               edge_index, edge_attr, single_step_mode=False)
                val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr, 
                                          single_step_mode=False)
                
                # Feature sensitivity
                avg_sens, max_sens, _ = self.test_feature_sensitivity(model, val_loader, edge_index, edge_attr)
                
                # Update scheduler
                scheduler.step()
                
                # Track history
                history['train_loss'].append(train_metrics['total'])
                history['val_loss'].append(val_metrics['total'])
                history['train_mse'].append(train_metrics['mse'])
                history['val_mse'].append(val_metrics['mse'])
                history['feature_sensitivity'].append(avg_sens)
                
                # Progress
                epoch_time = time.time() - start_time
                print(f"Epoch {self.config.single_step_epochs + epoch+1:2d}/{self.config.num_epochs}: "
                      f"Train={train_metrics['total']:.4f}, "
                      f"Val={val_metrics['total']:.4f}, "
                      f"Sens={avg_sens:.4f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time={epoch_time:.1f}s")
                
                # Early stopping
                if early_stopping(val_metrics['total'], model):
                    print(f"🛑 Early stopping at epoch {self.config.single_step_epochs + epoch+1}")
                    model.load_state_dict(early_stopping.best_model_state)
                    break
        
        # Final evaluation
        print(f"\n📊 Final evaluation...")
        final_train_metrics = self.validate(model, train_loader, criterion, edge_index, edge_attr, single_step_mode=False)
        final_val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr, single_step_mode=False)
        final_sens_avg, final_sens_max, sensitivities = self.test_feature_sensitivity(model, val_loader, edge_index, edge_attr)
        
        print(f"✅ Training complete!")
        print(f"   Final train loss: {final_train_metrics['total']:.4f}")
        print(f"   Final val loss: {final_val_metrics['total']:.4f}")
        print(f"   Feature sensitivity: {final_sens_avg:.4f} (avg), {final_sens_max:.4f} (max)")
        
        # Save model
        self.save_model(model, history, final_train_metrics, final_val_metrics, 
                       edge_index, edge_attr, sensitivities)
        
        # Generate plots
        self.plot_training_history(history)
        
        return model, history, edge_index, edge_attr
    
    def save_model(self, model, history, train_metrics, val_metrics, edge_index, edge_attr, sensitivities):
        """Save trained model"""
        
        model_data = {
            'experiment_id': self.experiment_id,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'normalizer': {
                'feature_scaler': self.normalizer.feature_scaler,
                'target_scaler': self.normalizer.target_scaler
            },
            'edge_index': edge_index.cpu(),
            'edge_attr': edge_attr.cpu(),
            'training_history': history,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'feature_sensitivities': sensitivities,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.log_dir / "spatiotemporal_model.pt"
        torch.save(model_data, model_path)
        
        print(f"💾 Model saved: {model_path}")
        
        # Save config as JSON for easy inspection
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE curves
        axes[0, 1].plot(epochs, history['train_mse'], 'g-', label='Train MSE')
        axes[0, 1].plot(epochs, history['val_mse'], 'orange', label='Val MSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].set_title('MSE Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Feature sensitivity
        axes[1, 0].plot(epochs, history['feature_sensitivity'], 'purple', label='Avg Sensitivity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Feature Sensitivity')
        axes[1, 0].set_title('Feature Sensitivity Over Time')
        axes[1, 0].grid(True)
        
        # Phase transition line
        if self.config.start_with_single_step:
            axes[1, 0].axvline(x=self.config.single_step_epochs, color='red', linestyle='--', alpha=0.7, label='Multi-step Phase')
            axes[1, 0].legend()
        
        # Summary
        axes[1, 1].axis('off')
        summary_text = f"""
Spatiotemporal Wave Model Training Summary:
• Experiment: {self.experiment_id}
• Final Train Loss: {history['train_loss'][-1]:.4f}
• Final Val Loss: {history['val_loss'][-1]:.4f}
• Final Sensitivity: {history['feature_sensitivity'][-1]:.4f}
• Sequence Length: {self.config.sequence_length}
• Prediction Horizon: {self.config.prediction_horizon}
• Hidden Dim: {self.config.hidden_dim}
• Spatial Layers: {self.config.num_spatial_layers}
• Temporal Layers: {self.config.num_temporal_layers}
• Use Spatial Attention: {self.config.use_spatial_attention}
• Use Temporal Attention: {self.config.use_temporal_attention}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Spatiotemporal Wave Model Training: {self.experiment_id}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.log_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training plots saved: {plot_path}")

# Evaluation utilities
class SpatioTemporalEvaluator:
    """Evaluator for the trained spatiotemporal model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        # Device setup with MPS support
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        
        print(f"📂 Loading spatiotemporal model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.config = checkpoint['config']
        self.normalizer = SimpleNormalizer()
        self.normalizer.feature_scaler = checkpoint['normalizer']['feature_scaler']
        self.normalizer.target_scaler = checkpoint['normalizer']['target_scaler']
        self.normalizer.fitted = True
        
        self.edge_index = checkpoint['edge_index'].to(self.device)
        self.edge_attr = checkpoint['edge_attr'].to(self.device)
        
        # Create and load model
        self.model = SpatioTemporalWaveGNN(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Sequence length: {self.config.sequence_length}")
        print(f"   Prediction horizon: {self.config.prediction_horizon}")
    
    def predict(self, input_sequence: torch.Tensor, multi_step: bool = True) -> torch.Tensor:
        """
        Make prediction with the model
        
        Args:
            input_sequence: [seq_len, num_nodes, features] or [batch_size, seq_len, num_nodes, features]
            multi_step: Whether to use multi-step prediction
        """
        
        if input_sequence.dim() == 3:
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
        
        input_sequence = input_sequence.to(self.device)
        
        # Normalize features
        batch_size, seq_len, num_nodes, num_features = input_sequence.size()
        input_flat = input_sequence.view(-1, num_features).cpu().numpy()
        input_norm = self.normalizer.transform_features(input_flat)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32, device=self.device)
        input_tensor = input_tensor.view(batch_size, seq_len, num_nodes, num_features)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(input_tensor, self.edge_index, self.edge_attr, multi_step=multi_step)
        
        # Denormalize predictions
        pred_flat = predictions.view(-1, predictions.size(-1)).cpu().numpy()
        pred_denorm = self.normalizer.inverse_transform_targets(pred_flat)
        predictions = torch.tensor(pred_denorm, dtype=torch.float32)
        predictions = predictions.view(batch_size, num_nodes, -1)
        
        return predictions.squeeze(0) if batch_size == 1 else predictions
    
    def evaluate_autoregressive(self, initial_sequence: torch.Tensor, num_steps: int = 10) -> Dict[str, Any]:
        """
        Evaluate autoregressive prediction capability
        
        Args:
            initial_sequence: [seq_len, num_nodes, features]
            num_steps: Number of autoregressive steps
        """
        
        print(f"🔄 Running autoregressive evaluation for {num_steps} steps...")
        
        # Setup
        current_sequence = initial_sequence.clone()
        predictions = []
        
        for step in range(num_steps):
            # Predict next step (single-step mode)
            pred = self.predict(current_sequence, multi_step=False)  # [num_nodes, 3]
            predictions.append(pred)
            
            # Update sequence for next prediction
            # Create new features by copying last timestep and updating wave variables
            new_features = current_sequence[-1].clone()  # [num_nodes, features]
            new_features[:, [3, 4, 5]] = pred  # Update SWH, MWD, MWP
            
            # Shift sequence and add new timestep
            current_sequence = torch.cat([current_sequence[1:], new_features.unsqueeze(0)], dim=0)
        
        # Stack predictions
        all_predictions = torch.stack(predictions, dim=0)  # [num_steps, num_nodes, 3]
        
        # Analyze predictions
        swh_preds = all_predictions[:, :, 0]  # [num_steps, num_nodes]
        mwd_preds = all_predictions[:, :, 1]
        mwp_preds = all_predictions[:, :, 2]
        
        # Check for stability
        swh_std = torch.std(swh_preds, dim=0).mean()  # Average std across nodes
        mwd_std = torch.std(mwd_preds, dim=0).mean()
        mwp_std = torch.std(mwp_preds, dim=0).mean()
        
        # Check for physical reasonableness
        swh_reasonable = torch.all(swh_preds >= 0) and torch.all(swh_preds <= 20)
        mwd_reasonable = torch.all(mwd_preds >= 0) and torch.all(mwd_preds <= 360)
        mwp_reasonable = torch.all(mwp_preds >= 1) and torch.all(mwp_preds <= 25)
        
        results = {
            'predictions': all_predictions,
            'stability': {
                'swh_std': swh_std.item(),
                'mwd_std': mwd_std.item(),
                'mwp_std': mwp_std.item()
            },
            'physical_bounds': {
                'swh_reasonable': swh_reasonable.item(),
                'mwd_reasonable': mwd_reasonable.item(),
                'mwp_reasonable': mwp_reasonable.item()
            },
            'final_ranges': {
                'swh': (swh_preds[-1].min().item(), swh_preds[-1].max().item()),
                'mwd': (mwd_preds[-1].min().item(), mwd_preds[-1].max().item()),
                'mwp': (mwp_preds[-1].min().item(), mwp_preds[-1].max().item())
            }
        }
        
        print(f"✅ Autoregressive evaluation complete!")
        print(f"   Stability (std): SWH={swh_std:.3f}, MWD={mwd_std:.1f}, MWP={mwp_std:.3f}")
        print(f"   Physical bounds: SWH={swh_reasonable}, MWD={mwd_reasonable}, MWP={mwp_reasonable}")
        
        return results

def main():
    """Main function to run training and evaluation"""
    
    print("🌊 SPATIOTEMPORAL WAVE FORECASTING MODEL")
    print("=" * 60)
    
    # Configuration
    config = SpatioTemporalConfig(
        # Temporal settings
        sequence_length=6,
        prediction_horizon=4,
        
        # Architecture
        hidden_dim=256,
        temporal_hidden_dim=128,
        num_spatial_layers=6,
        num_temporal_layers=2,
        
        # Attention
        use_spatial_attention=True,
        use_temporal_attention=True,
        
        # Training
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-4,
        
        # Curriculum learning
        start_with_single_step=True,
        single_step_epochs=15,
        
        # Data
        max_training_sequences=1000
    )
    
    print(f"🔧 Configuration:")
    print(f"   Sequence length: {config.sequence_length}")
    print(f"   Prediction horizon: {config.prediction_horizon}")
    print(f"   Hidden dimension: {config.hidden_dim}")
    print(f"   Spatial layers: {config.num_spatial_layers}")
    print(f"   Temporal layers: {config.num_temporal_layers}")
    print(f"   Use attention: Spatial={config.use_spatial_attention}, Temporal={config.use_temporal_attention}")
    
    # Train model
    trainer = SpatioTemporalTrainer(config)
    model, history, edge_index, edge_attr = trainer.train()
    
    # Evaluate model
    model_path = trainer.log_dir / "spatiotemporal_model.pt"
    evaluator = SpatioTemporalEvaluator(model_path)
    
    # Test autoregressive prediction
    print(f"\n🔍 Testing autoregressive prediction...")
    
    # Create test sequence (using random data for demonstration)
    test_sequence = torch.randn(config.sequence_length, 1170, 11)  # Assuming 1170 nodes
    
    # Run autoregressive evaluation
    results = evaluator.evaluate_autoregressive(test_sequence, num_steps=10)
    
    print(f"\n🎉 Training and evaluation complete!")
    print(f"   Model saved to: {model_path}")
    print(f"   Experiment directory: {trainer.log_dir}")

# if __name__ == "__main__":
#     # main()
#     config = SpatioTemporalConfig(
#         # Temporal settings
#         sequence_length=6,
#         prediction_horizon=4,
        
#         # Architecture
#         hidden_dim=256,
#         temporal_hidden_dim=128,
#         num_spatial_layers=6,
#         num_temporal_layers=2,
        
#         # Attention
#         use_spatial_attention=True,
#         use_temporal_attention=True,
        
#         # Training
#         num_epochs=50,
#         batch_size=8,
#         learning_rate=1e-4,
        
#         # Curriculum learning
#         start_with_single_step=True,
#         single_step_epochs=15,
        
#         # Data
#         max_training_sequences=1000
#     )
#     model_path = 'xperiments/spatiotemporal_20250702_235533/spatiotemporal_model.pt'
#     evaluator = SpatioTemporalEvaluator(model_path)
#     test_sequence = torch.randn(config.sequence_length, 1170, 11)
#     results = evaluator.evaluate_autoregressive(test_sequence, num_steps=10)
