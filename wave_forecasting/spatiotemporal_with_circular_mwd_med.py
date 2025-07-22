#!/usr/bin/env python3
"""
Spatiotemporal Wave Model with Circular MWD Fixes
Integrates circular normalization and loss for wave direction
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
import xarray as xr

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import existing components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import DataConfig, MeshConfig

# Import the MWD fixes
from mwd_circular_fixes import VariableSpecificNormalizer, CircularLoss, evaluate_model_with_circular_metrics

@dataclass
class SpatioTemporalConfig:
    """Configuration for spatiotemporal wave model with circular MWD"""
    
    # Temporal parameters
    sequence_length: int = 6           
    prediction_horizon: int = 4        
    
    # Model architecture
    input_features: int = 11
    hidden_dim: int = 320
    temporal_hidden_dim: int = 128
    num_spatial_layers: int = 6        
    num_temporal_layers: int = 4       
    
    # Attention
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    num_attention_heads: int = 8
    
    # Regularization
    dropout: float = 0.15
    spatial_dropout: float = 0.1
    temporal_dropout: float = 0.1
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 8               
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Circular MWD loss weights
    mse_weight: float = 1.0
    circular_weight: float = 1.0
    physics_loss_weight: float = 0.2
    temporal_consistency_weight: float = 0.1
    
    # Data
    validation_split: float = 0.2
    max_training_sequences: int = 1000  
    
    # Curriculum learning
    start_with_single_step: bool = True
    single_step_epochs: int = 15

# Copy existing classes (SpatialAttention, TemporalAttention, EnhancedMessageLayer)
# [Previous classes remain the same - just copying for completeness]

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

class SpatioTemporalWaveGNN(nn.Module):
    """Spatiotemporal model with circular MWD output"""
    
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
        
        # Prediction heads - MODIFIED for circular MWD
        self.single_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 4)  # [SWH, MWD_cos, MWD_sin, MWP]
        )
        
        self.multi_step_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 4 * config.prediction_horizon)  # 4 features × horizon
        )
        
        # Physics constraints
        self.physics_layer = nn.Linear(4 * config.prediction_horizon, 4 * config.prediction_horizon)
        
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
        Forward pass with circular MWD output
        Returns [SWH, MWD_cos, MWD_sin, MWP] format
        """
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        # Process each timestep spatially
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            h_t = self.feature_encoder(x_t)
            
            # Spatial processing for each sample in batch
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]
                
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
        spatial_sequence = torch.stack(spatial_outputs, dim=1)
        
        # Temporal processing for each node
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]
            
            # LSTM
            lstm_out, _ = self.temporal_encoder(node_sequence)
            
            # Temporal attention
            if self.config.use_temporal_attention:
                lstm_out = self.temporal_attention(lstm_out)
            
            # Use last timestep
            final_state = lstm_out[:, -1, :]
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)
        
        # Generate predictions
        if multi_step:
            predictions = self.multi_step_head(temporal_features)
            # Apply physics constraints
            physics_adjustment = self.physics_layer(predictions)
            predictions = predictions + 0.1 * physics_adjustment
        else:
            predictions = self.single_step_head(temporal_features)
        
        return predictions

class SpatioTemporalTrainer:
    """Trainer with circular MWD handling"""
    
    def __init__(self, config: SpatioTemporalConfig):
        self.config = config
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Use variable-specific normalizer
        self.normalizer = StandardScaler()  # For features
        self.target_normalizer = VariableSpecificNormalizer()  # For circular MWD targets
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"spatiotemporal_circular_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Spatiotemporal Training with Circular MWD: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup data with circular MWD handling"""
        
        print("📊 Setting up spatiotemporal dataset with circular MWD...")
        
        # Load data components (same as before)
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)

        # Multi-year (what you need):
        all_atmo_data = []
        all_wave_data = []

        for year in [2017, 2018, 2019, 2020, 2021, 2022]:
            for month in range(1, 13):  # All 12 months
                try:
                    atmo, waves = era5_manager.load_month_data(year, month)
                    all_atmo_data.append(atmo)
                    all_wave_data.append(waves)
                except:
                    print(f"Skipping {year}-{month:02d}")

        # Concatenate all data
        era5_atmo = xr.concat(all_atmo_data, dim='valid_time')
        era5_waves = xr.concat(all_wave_data, dim='valid_time')

        gebco_data = gebco_manager.load_bathymetry()
        
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create dataset
        dataset = SpatioTemporalDataset(mesh_loader, self.config, start_time=0, end_time=100)
        
        # Fit normalizers
        print("🔧 Fitting circular normalizers...")
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(100, len(dataset)), 10):
            sample = dataset[i]
            features = sample['input'].numpy()
            targets = sample['single_step_target'].numpy()  # [num_nodes, 3] - SWH, MWD, MWP
            
            features_flat = features.reshape(-1, features.shape[-1])
            sample_features.append(features_flat)
            sample_targets.append(targets)
        
        if sample_features:
            all_features = np.vstack(sample_features)
            all_targets = np.vstack(sample_targets)
            
            # Fit normalizers
            self.normalizer.fit(all_features)
            self.target_normalizer.fit(all_targets)  # This handles circular MWD
        
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
        
        print(f"✅ Data setup complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Nodes: {len(region_indices)}")
        print(f"   Edges: {edge_index.shape[1]}")
        
        return train_loader, val_loader, edge_index, edge_attr
    
    def train_epoch(self, model, train_loader, criterion, optimizer, edge_index, edge_attr, single_step_mode: bool = True):
        """Train one epoch with circular MWD"""
        
        model.train()
        epoch_losses = {'total': [], 'swh': [], 'mwd_circular': [], 'mwd_angular': [], 'mwp': [], 'physics': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs = batch['input'].to(self.device)
            
            if single_step_mode:
                targets = batch['single_step_target'].to(self.device)  # [batch_size, num_nodes, 3]
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
                targets_norm = self.target_normalizer.transform_targets(targets_flat)  # [N, 4] - SWH, MWD_cos, MWD_sin, MWP
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
            else:
                # Multi-step: handle multiple timesteps
                batch_size, num_nodes, total_features = targets.size()
                horizon = total_features // 3
                targets_reshaped = targets.view(batch_size, num_nodes, horizon, 3)
                targets_flat = targets_reshaped.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)  # [N, 4]
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, horizon * 4)
            
            # Forward pass
            predictions = model(inputs, edge_index, edge_attr, multi_step=not single_step_mode)
            
            # Compute circular loss
            loss_dict = criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                if f'{key}_loss' in loss_dict:
                    epoch_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    def train(self):
        """Main training loop with circular MWD"""
        
        print(f"🚀 Starting spatiotemporal training with circular MWD...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        
        # Create model and circular loss
        model = SpatioTemporalWaveGNN(self.config).to(self.device)
        criterion = CircularLoss(
            mse_weight=self.config.mse_weight,
            circular_weight=self.config.circular_weight,
            physics_weight=self.config.physics_loss_weight
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f"✅ Model created with circular MWD support:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Output format: [SWH, MWD_cos, MWD_sin, MWP]")
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'mwd_circular': [], 'mwd_angular': []}
        
        # Single-step training
        print(f"\n📈 Single-step training with circular MWD ({self.config.single_step_epochs} epochs)")
        
        for epoch in range(self.config.single_step_epochs):
            start_time = time.time()
            
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, edge_index, edge_attr, single_step_mode=True)
            
            # Track history
            history['train_loss'].append(train_metrics['total'])
            history['mwd_circular'].append(train_metrics.get('mwd_circular', 0))
            history['mwd_angular'].append(train_metrics.get('mwd_angular', 0))
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{self.config.single_step_epochs}: "
                  f"Total={train_metrics['total']:.4f}, "
                  f"SWH={train_metrics.get('swh', 0):.4f}, "
                  f"MWD_circ={train_metrics.get('mwd_circular', 0):.4f}, "
                  f"MWD_ang={train_metrics.get('mwd_angular', 0):.4f}, "
                  f"MWP={train_metrics.get('mwp', 0):.4f}, "
                  f"Time={epoch_time:.1f}s")
        
        # Save model
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
        
        model_path = self.log_dir / "spatiotemporal_circular_model.pt"
        torch.save(model_data, model_path)
        
        print(f"✅ Training complete!")
        print(f"💾 Model saved: {model_path}")
        
        return model, history, edge_index, edge_attr

# Copy the SpatioTemporalDataset class from original
class SpatioTemporalDataset(Dataset):
    """Dataset for spatiotemporal sequences"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: SpatioTemporalConfig, 
                 start_time: int = 0, end_time: int = 100):
        self.mesh_loader = mesh_loader
        self.config = config
        self.sequences = []
        
        print(f"Building spatiotemporal dataset from timesteps {start_time} to {end_time}...")
        
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
        
        print(f"Created {len(self.sequences)} spatiotemporal sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def main():
    """Main function to train spatiotemporal model with circular MWD"""
    
    print("🌊 SPATIOTEMPORAL WAVE MODEL WITH CIRCULAR MWD")
    print("=" * 60)
    
    # Configuration
    config = SpatioTemporalConfig(
        # Temporal settings
        sequence_length=6,
        prediction_horizon=4,
        
        # Architecture
        hidden_dim=320,
        temporal_hidden_dim=128,
        num_spatial_layers=6,
        num_temporal_layers=2,
        
        # Attention
        use_spatial_attention=True,
        use_temporal_attention=True,
        
        # Training
        num_epochs=120,  # Shorter for testing
        batch_size=8,
        learning_rate=1e-4,
        
        # Circular MWD loss weights
        mse_weight=1.0,
        circular_weight=1.0,
        physics_loss_weight=0.2,
        
        # Curriculum learning
        start_with_single_step=True,
        single_step_epochs=120,
        
        # Data
        max_training_sequences=1000
    )
    
    print(f"🔧 Configuration:")
    print(f"   Circular MWD enabled: True")
    print(f"   MWD loss weights: Circular={config.circular_weight}, Physics={config.physics_loss_weight}")
    print(f"   Expected output improvement: 127° → ~20-40° MWD RMSE")
    
    # Train model
    trainer = SpatioTemporalTrainer(config)
    model, history, edge_index, edge_attr = trainer.train()
    
    print(f"\n🎉 Circular MWD training complete!")
    print(f"   Next step: Run evaluation to compare MWD performance")
    print(f"   Expected: Dramatic improvement in wave direction accuracy")

if __name__ == "__main__":
    main()
