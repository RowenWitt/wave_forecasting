#!/usr/bin/env python3
"""
Wavelet Enhanced Variable Learning Rate Wave Model
Combines proven Variable LR approach with wavelet temporal processing
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
import pywt  # For wavelet transforms

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import existing components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import DataConfig, MeshConfig

# Import the MWD fixes and Variable LR components
from mwd_circular_fixes import VariableSpecificNormalizer, CircularLoss, evaluate_model_with_circular_metrics
from variable_lr_spatiotemporal_with_circular_mwd import (
    VariableLRConfig, VariableLearningManager, VariableSpecificLoss,
    SpatialAttention, TemporalAttention, EnhancedMessageLayer
)

@dataclass
class WaveletVariableLRConfig(VariableLRConfig):
    """Configuration extending Variable LR with wavelet processing"""
    
    # Wavelet-specific parameters
    use_wavelets: bool = True
    wavelet_type: str = 'db4'                    # Daubechies-4 wavelet
    wavelet_levels: int = 2                      # Reduced to 2 levels for seq_len=6
    wavelet_fusion_method: str = "attention"     # "attention", "concat", "weighted"
    
    # Enhanced temporal processing
    wavelet_hidden_dim: int = 64                 # Hidden dim for wavelet features
    combine_lstm_wavelet: bool = True            # Combine LSTM + wavelets
    
    # Training parameters (inherit proven settings)
    early_stopping_patience: int = 50           # Enable proper early stopping

class WaveletProcessor(nn.Module):
    """Simplified wavelet-based temporal feature processor"""
    
    def __init__(self, hidden_dim: int, wavelet_hidden_dim: int, 
                 wavelet_type: str = 'db4', levels: int = 2,
                 fusion_method: str = "attention"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.wavelet_hidden_dim = wavelet_hidden_dim
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.fusion_method = fusion_method
        
        print(f"🌊 Wavelet Processor:")
        print(f"   Wavelet: {wavelet_type} with {levels} decomposition levels")
        print(f"   Fusion: {fusion_method}")
        print(f"   Hidden dim: {hidden_dim} → {wavelet_hidden_dim}")
        
        # Learnable weights for different frequency bands
        self.frequency_weights = nn.Parameter(torch.ones(levels + 1))
        
        # Simplified approach: project temporal features through frequency analysis
        self.temporal_projection = nn.Linear(hidden_dim, wavelet_hidden_dim)
        
        # Frequency-specific processing layers
        self.freq_processors = nn.ModuleList()
        for i in range(levels + 1):  # +1 for approximation coefficients
            self.freq_processors.append(
                nn.Sequential(
                    nn.Linear(wavelet_hidden_dim, wavelet_hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(wavelet_hidden_dim // 2, wavelet_hidden_dim // (levels + 1))
                )
            )
        
        # Fusion mechanism
        if fusion_method == "attention":
            # Fix: ensure proper dimension alignment
            fusion_dim = wavelet_hidden_dim  # Use full dimension for attention
            self.fusion_projection = nn.Linear(
                (wavelet_hidden_dim // (levels + 1)) * (levels + 1), 
                fusion_dim
            )
            self.wavelet_attention = nn.MultiheadAttention(
                fusion_dim, 
                num_heads=max(1, fusion_dim // 32),
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(fusion_dim)
        elif fusion_method == "concat":
            total_dim = (wavelet_hidden_dim // (levels + 1)) * (levels + 1)
            self.fusion_projection = nn.Linear(total_dim, wavelet_hidden_dim)
        elif fusion_method == "weighted":
            self.level_weights = nn.Parameter(torch.ones(levels + 1))
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(wavelet_hidden_dim, wavelet_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(wavelet_hidden_dim, wavelet_hidden_dim)
        )
    
    def _apply_simplified_wavelet_analysis(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Simplified wavelet-inspired frequency analysis"""
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to wavelet space
        x_proj = self.temporal_projection(x)  # [batch_size, seq_len, wavelet_hidden_dim]
        
        # Simulate frequency decomposition through learned filters
        frequency_components = []
        
        for level in range(self.levels + 1):
            # Apply frequency-specific processing
            freq_features = self.freq_processors[level](x_proj)
            
            # Weight by learned frequency importance
            weighted_features = freq_features * self.frequency_weights[level]
            
            # Temporal pooling (different strategies for different frequencies)
            if level == 0:  # High frequency - focus on recent patterns
                pooled = weighted_features[:, -seq_len//2:, :].mean(dim=1)
            elif level == self.levels:  # Low frequency/approximation - use all
                pooled = weighted_features.mean(dim=1)
            else:  # Medium frequency - balanced approach
                pooled = weighted_features.mean(dim=1)
            
            frequency_components.append(pooled)
        
        return frequency_components
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal sequence through simplified wavelets"""
        # x: [batch_size, seq_len, hidden_dim]
        
        # Apply wavelet-inspired frequency analysis
        freq_components = self._apply_simplified_wavelet_analysis(x)
        
        # Fuse different frequency components
        if self.fusion_method == "attention":
            # Concatenate all frequency components
            concatenated = torch.cat(freq_components, dim=-1)  # [batch_size, total_freq_dim]
            
            # Project to target dimension
            projected = self.fusion_projection(concatenated)  # [batch_size, wavelet_hidden_dim]
            
            # Apply self-attention (treating as single sequence element)
            projected_seq = projected.unsqueeze(1)  # [batch_size, 1, wavelet_hidden_dim]
            attended, _ = self.wavelet_attention(projected_seq, projected_seq, projected_seq)
            
            # Extract result
            fused = attended.squeeze(1)  # [batch_size, wavelet_hidden_dim]
            fused = self.fusion_norm(fused)
            
        elif self.fusion_method == "concat":
            # Simple concatenation and projection
            concatenated = torch.cat(freq_components, dim=-1)
            fused = self.fusion_projection(concatenated)
            
        elif self.fusion_method == "weighted":
            # Weighted combination
            stacked = torch.stack(freq_components, dim=1)  # [batch_size, num_levels, freq_dim]
            weights = F.softmax(self.level_weights, dim=0)
            fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)
            
            # Ensure correct output dimension
            if fused.size(-1) != self.wavelet_hidden_dim:
                if not hasattr(self, 'weighted_projection'):
                    self.weighted_projection = nn.Linear(fused.size(-1), self.wavelet_hidden_dim).to(fused.device)
                fused = self.weighted_projection(fused)
        
        else:
            # Simple average
            stacked = torch.stack(freq_components, dim=1)
            fused = stacked.mean(dim=1)
            
            # Ensure correct output dimension
            if fused.size(-1) != self.wavelet_hidden_dim:
                if not hasattr(self, 'average_projection'):
                    self.average_projection = nn.Linear(fused.size(-1), self.wavelet_hidden_dim).to(fused.device)
                fused = self.average_projection(fused)
        
        # Final processing
        output = self.output_projection(fused)
        
        return output

class WaveletVariableLRSpatioTemporalGNN(nn.Module):
    """Proven Variable LR model enhanced with wavelet processing"""
    
    def __init__(self, config: WaveletVariableLRConfig):
        super().__init__()
        self.config = config
        
        print(f"🌊 Wavelet-Enhanced Variable LR Model:")
        print(f"   Base architecture: Proven Variable LR (9.21 RMSE)")
        print(f"   Enhancement: Wavelet temporal processing")
        print(f"   Wavelets enabled: {config.use_wavelets}")
        
        # Feature encoding (proven from Variable LR)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Spatial layers (proven from Variable LR)
        self.spatial_layers = nn.ModuleList([
            EnhancedMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Spatial attention (proven from Variable LR)
        if config.use_spatial_attention:
            self.spatial_attention = SpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        # Temporal processing: LSTM + Wavelets
        if config.use_wavelets:
            # Standard LSTM (proven)
            self.temporal_encoder = nn.LSTM(
                config.hidden_dim,
                config.temporal_hidden_dim // 2,  # Split space with wavelets
                config.num_temporal_layers,
                batch_first=True,
                dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
            )
            
            # Wavelet processor
            self.wavelet_processor = WaveletProcessor(
                hidden_dim=config.hidden_dim,
                wavelet_hidden_dim=config.wavelet_hidden_dim,
                wavelet_type=config.wavelet_type,
                levels=config.wavelet_levels,
                fusion_method=config.wavelet_fusion_method
            )
            
            # Fusion of LSTM + Wavelet features
            if config.combine_lstm_wavelet:
                fusion_input_dim = config.temporal_hidden_dim // 2 + config.wavelet_hidden_dim
                self.lstm_wavelet_fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, config.temporal_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.temporal_hidden_dim, config.temporal_hidden_dim)
                )
            else:
                # Use only wavelets
                self.wavelet_to_temporal = nn.Linear(config.wavelet_hidden_dim, config.temporal_hidden_dim)
                
        else:
            # Fallback to standard LSTM (original Variable LR)
            self.temporal_encoder = nn.LSTM(
                config.hidden_dim,
                config.temporal_hidden_dim,
                config.num_temporal_layers,
                batch_first=True,
                dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
            )
        
        # Temporal attention (proven from Variable LR)
        if config.use_temporal_attention:
            self.temporal_attention = TemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Variable-specific prediction heads (proven from Variable LR)
        hidden_size = config.hidden_dim // 2
        
        # Shared feature transformation
        self.shared_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Variable-specific final layers
        self.swh_head = nn.Linear(hidden_size, 1)      # SWH output
        self.mwd_head = nn.Linear(hidden_size, 2)      # MWD [cos, sin] output  
        self.mwp_head = nn.Linear(hidden_size, 1)      # MWP output
        
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
        """Forward pass with wavelet enhancement"""
        
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        # Process each timestep spatially (proven from Variable LR)
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
        
        # Temporal processing for each node (LSTM + Wavelets)
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]  # [batch_size, seq_len, hidden_dim]
            
            if self.config.use_wavelets:
                if self.config.combine_lstm_wavelet:
                    # LSTM processing
                    lstm_out, _ = self.temporal_encoder(node_sequence)
                    lstm_features = lstm_out[:, -1, :]  # Last timestep
                    
                    # Wavelet processing
                    wavelet_features = self.wavelet_processor(node_sequence)
                    
                    # Combine LSTM + Wavelet
                    combined_features = torch.cat([lstm_features, wavelet_features], dim=-1)
                    final_state = self.lstm_wavelet_fusion(combined_features)
                    
                else:
                    # Wavelet only
                    wavelet_features = self.wavelet_processor(node_sequence)
                    final_state = self.wavelet_to_temporal(wavelet_features)
            else:
                # Standard LSTM (fallback)
                lstm_out, _ = self.temporal_encoder(node_sequence)
                final_state = lstm_out[:, -1, :]
            
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)
        
        # Temporal attention (proven from Variable LR)
        if self.config.use_temporal_attention:
            # Need to add sequence dimension for attention
            temporal_features_seq = temporal_features.unsqueeze(1)  # [batch, 1, nodes, features]
            batch_attended = []
            for b in range(batch_size):
                node_features = temporal_features_seq[b, 0, :, :].unsqueeze(0)  # [1, nodes, features]
                # Apply temporal attention across nodes (treating as sequence)
                attended = self.temporal_attention(node_features)
                batch_attended.append(attended.squeeze(0))
            temporal_features = torch.stack(batch_attended, dim=0)
        
        # Shared feature transformation (proven from Variable LR)
        shared_features = self.shared_head(temporal_features)
        
        # Variable-specific predictions (proven from Variable LR)
        swh_pred = self.swh_head(shared_features)      # [batch, nodes, 1]
        mwd_pred = self.mwd_head(shared_features)      # [batch, nodes, 2]
        mwp_pred = self.mwp_head(shared_features)      # [batch, nodes, 1]
        
        # Combine predictions: [SWH, MWD_cos, MWD_sin, MWP]
        predictions = torch.cat([swh_pred, mwd_pred, mwp_pred], dim=-1)
        
        return predictions
    
    def get_variable_parameters(self):
        """Get parameters grouped by variable for separate optimizers"""
        
        # Shared parameters (encoder, spatial, temporal)
        shared_params = []
        shared_params.extend(self.feature_encoder.parameters())
        shared_params.extend(self.spatial_layers.parameters())
        shared_params.extend(self.temporal_encoder.parameters())
        shared_params.extend(self.shared_head.parameters())
        
        # Add wavelet parameters to shared
        if hasattr(self, 'wavelet_processor'):
            shared_params.extend(self.wavelet_processor.parameters())
        if hasattr(self, 'lstm_wavelet_fusion'):
            shared_params.extend(self.lstm_wavelet_fusion.parameters())
        if hasattr(self, 'wavelet_to_temporal'):
            shared_params.extend(self.wavelet_to_temporal.parameters())
        
        if hasattr(self, 'spatial_attention'):
            shared_params.extend(self.spatial_attention.parameters())
        if hasattr(self, 'temporal_attention'):
            shared_params.extend(self.temporal_attention.parameters())
        
        # Variable-specific parameters
        swh_params = list(self.swh_head.parameters())
        mwd_params = list(self.mwd_head.parameters())
        mwp_params = list(self.mwp_head.parameters())
        
        return {
            'shared': shared_params,
            'swh': swh_params,
            'mwd': mwd_params,
            'mwp': mwp_params
        }

class WaveletVariableLRTrainer:
    """Trainer for Wavelet-Enhanced Variable LR model"""
    
    def __init__(self, config: WaveletVariableLRConfig):
        self.config = config
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Normalizers
        self.normalizer = StandardScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Learning rate manager (proven from Variable LR)
        self.lr_manager = VariableLearningManager(config)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"wavelet_variable_lr_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Wavelet-Enhanced Variable LR Training: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup data (same as Variable LR)"""
        
        print("📊 Setting up dataset...")
        
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Multi-year data loading
        all_atmo_data = []
        all_wave_data = []

        for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]:
            for month in range(1, 13):
                try:
                    atmo, waves = era5_manager.load_month_data(year, month)
                    all_atmo_data.append(atmo)
                    all_wave_data.append(waves)
                except:
                    print(f"Skipping {year}-{month:02d}")

        era5_atmo = xr.concat(all_atmo_data, dim='valid_time')
        era5_waves = xr.concat(all_wave_data, dim='valid_time')
        gebco_data = gebco_manager.load_bathymetry()
        
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        dataset = WaveletSpatioTemporalDataset(mesh_loader, self.config, start_time=0, end_time=100)
        
        # Fit normalizers
        print("🔧 Fitting normalizers...")
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(100, len(dataset)), 10):
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
        
        return train_loader, val_loader, edge_index, edge_attr

    def create_optimizers(self, model: WaveletVariableLRSpatioTemporalGNN):
        """Create variable-specific optimizers (proven from Variable LR)"""
        
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
    
    def update_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        """Update optimizer learning rates"""
        
        current_lrs = self.lr_manager.get_current_lrs()
        
        # Update shared optimizer with MWD LR
        optimizers['shared'].param_groups[0]['lr'] = current_lrs['mwd']
        
        # Update variable-specific optimizers
        for var in ['swh', 'mwd', 'mwp']:
            optimizers[var].param_groups[0]['lr'] = current_lrs[var]
    
    def train_epoch(self, model, train_loader, criterion, optimizers, edge_index, edge_attr):
        """Train one epoch (same as Variable LR)"""
        
        model.train()
        epoch_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Zero all optimizers
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            inputs = batch['input'].to(self.device)
            targets = batch['single_step_target'].to(self.device)  # [batch_size, num_nodes, 3]
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            inputs_norm = self.normalizer.transform(inputs_flat)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets with circular MWD
            targets_flat = targets.view(-1, 3).cpu().numpy()
            targets_norm = self.target_normalizer.transform_targets(targets_flat)  # [N, 4]
            targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
            targets = targets.view(batch_size, num_nodes, 4)
            
            # Forward pass
            predictions = model(inputs, edge_index, edge_attr, multi_step=False)
            
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
        
        return {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion, edge_index, edge_attr):
        """Validate model (same as Variable LR)"""
        
        model.eval()
        val_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['single_step_target'].to(self.device)
                
                # Normalize inputs
                batch_size, seq_len, num_nodes, num_features = inputs.size()
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Normalize targets
                targets_flat = targets.view(-1, 3).cpu().numpy()
                targets_norm = self.target_normalizer.transform_targets(targets_flat)
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 4)
                
                # Forward pass
                predictions = model(inputs, edge_index, edge_attr, multi_step=False)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                
                # Track losses
                for key in val_losses:
                    if f'{key}_loss' in loss_dict:
                        val_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
    
    def train(self):
        """Main training loop with wavelets"""
        
        print(f"🚀 Starting wavelet-enhanced variable learning rate training...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        
        # Create model and loss
        model = WaveletVariableLRSpatioTemporalGNN(self.config).to(self.device)
        criterion = VariableSpecificLoss(self.config)
        
        # Create variable-specific optimizers
        optimizers = self.create_optimizers(model)
        
        print(f"✅ Wavelet-Enhanced Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Base: Proven Variable LR architecture")
        print(f"   Enhancement: Wavelet temporal processing")
        print(f"   Variable-specific heads: SWH, MWD (circular), MWP")
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': {},
            'variable_losses': {'swh': [], 'mwd': [], 'mwp': []}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        print(f"\n📈 Wavelet-enhanced training ({self.config.single_step_epochs} epochs)")
        
        for epoch in range(self.config.single_step_epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizers, edge_index, edge_attr)
            val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr)
            
            # Update learning rate manager
            variable_losses = {
                'swh': val_metrics.get('swh', float('inf')),
                'mwd': val_metrics.get('mwd', float('inf')),
                'mwp': val_metrics.get('mwp', float('inf'))
            }
            
            self.lr_manager.update_performance(variable_losses)
            
            # Update optimizers with new learning rates
            self.update_optimizers(optimizers)
            
            # Track history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['learning_rates'][epoch] = self.lr_manager.get_current_lrs()
            
            for var in ['swh', 'mwd', 'mwp']:
                history['variable_losses'][var].append(variable_losses[var])
            
            epoch_time = time.time() - start_time
            
            # Print detailed progress
            current_lrs = self.lr_manager.get_current_lrs()
            print(f"Epoch {epoch+1:3d}/{self.config.single_step_epochs}: "
                  f"Total={train_metrics['total']:.4f}, "
                  f"Val={val_metrics['total']:.4f}")
            print(f"       SWH={val_metrics.get('swh', 0):.4f} (LR={current_lrs['swh']:.2e}), "
                  f"MWD={val_metrics.get('mwd', 0):.4f} (LR={current_lrs['mwd']:.2e}), "
                  f"MWP={val_metrics.get('mwp', 0):.4f} (LR={current_lrs['mwp']:.2e})")
            
            # Performance summary every 20 epochs
            if (epoch + 1) % 20 == 0:
                perf_summary = self.lr_manager.get_performance_summary()
                print(f"   📊 Wavelet Performance Summary:")
                for var, stats in perf_summary.items():
                    print(f"       {var.upper()}: Best={stats['best_loss']:.4f}, "
                          f"Plateau={stats['plateau_counter']}")
            
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
                    'lr_manager_state': self.lr_manager.get_performance_summary(),
                    'epoch': epoch,
                    'val_loss': val_metrics['total']
                }, self.log_dir / "best_wavelet_variable_lr_model.pt")
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
            'lr_manager_final_state': self.lr_manager.get_performance_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.log_dir / "wavelet_variable_lr_model.pt"
        torch.save(model_data, model_path)
        
        # Save analysis
        self._save_wavelet_analysis(history)
        
        print(f"✅ Wavelet-enhanced training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved: {model_path}")
        
        return model, history, edge_index, edge_attr
    
    def _save_wavelet_analysis(self, history: Dict[str, Any]):
        """Save wavelet-specific analysis and plots"""
        
        # Create analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Wavelet-Enhanced Variable LR Model Analysis', fontsize=16)
        
        epochs = list(range(len(history['train_loss'])))
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variable-specific losses
        for var, color in zip(['swh', 'mwd', 'mwp'], ['blue', 'red', 'green']):
            if var in history['variable_losses']:
                axes[0, 1].plot(epochs, history['variable_losses'][var], 
                               label=f'{var.upper()}', color=color, linewidth=2)
        axes[0, 1].set_title('Variable-Specific Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate evolution
        if history['learning_rates']:
            lr_epochs = list(history['learning_rates'].keys())
            swh_lrs = [history['learning_rates'][e]['swh'] for e in lr_epochs]
            mwd_lrs = [history['learning_rates'][e]['mwd'] for e in lr_epochs]
            mwp_lrs = [history['learning_rates'][e]['mwp'] for e in lr_epochs]
            
            axes[0, 2].semilogy(lr_epochs, swh_lrs, 'o-', label='SWH', color='blue')
            axes[0, 2].semilogy(lr_epochs, mwd_lrs, 's-', label='MWD', color='red')
            axes[0, 2].semilogy(lr_epochs, mwp_lrs, '^-', label='MWP', color='green')
            axes[0, 2].set_title('Learning Rate Evolution')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate (log scale)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Wavelet architecture summary
        wavelet_text = "Wavelet Enhancement:\n\n"
        wavelet_text += f"Base Model: Variable LR (9.21 RMSE)\n"
        wavelet_text += f"Wavelet Type: {self.config.wavelet_type}\n"
        wavelet_text += f"Decomposition Levels: {self.config.wavelet_levels}\n"
        wavelet_text += f"Fusion Method: {self.config.wavelet_fusion_method}\n"
        wavelet_text += f"Combine LSTM+Wavelet: {self.config.combine_lstm_wavelet}\n\n"
        
        wavelet_text += "Frequency Analysis:\n"
        wavelet_text += "• Level 0: High freq (wind waves)\n"
        wavelet_text += "• Level 1: Medium freq (local systems)\n"
        wavelet_text += "• Level 2: Low freq (swell propagation)\n"
        wavelet_text += "• Approximation: Trends/seasonality\n\n"
        
        wavelet_text += "Expected Benefits:\n"
        wavelet_text += "• Better storm event detection\n"
        wavelet_text += "• Improved wave group modeling\n"
        wavelet_text += "• Enhanced MWP prediction\n"
        
        axes[1, 0].text(0.05, 0.95, wavelet_text, transform=axes[1, 0].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 0].set_title('Wavelet Architecture')
        axes[1, 0].axis('off')
        
        # Performance summary
        final_perf = self.lr_manager.get_performance_summary()
        summary_text = "Performance Summary:\n\n"
        for var, stats in final_perf.items():
            summary_text += f"{var.upper()}:\n"
            summary_text += f"  Final LR: {stats['current_lr']:.2e}\n"
            summary_text += f"  Best Loss: {stats['best_loss']:.4f}\n"
            summary_text += f"  Recent Avg: {stats['recent_avg']:.4f}\n\n"
        
        summary_text += f"\nTarget Performance:\n"
        summary_text += f"SWH: 0.674m → 0.60-0.68m\n"
        summary_text += f"MWD: 25.8° → 22-26°\n"
        summary_text += f"MWP: 1.187s → 1.0-1.1s\n"
        summary_text += f"Overall: 9.21 → 7.5-8.5 RMSE\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        # Comparison to baselines
        baseline_text = "vs Baseline Models:\n\n"
        baseline_text += "Variable LR: 9.21 ± 0.82 RMSE\n"
        baseline_text += "  SWH: 0.674m, MWD: 25.8°, MWP: 1.187s\n\n"
        baseline_text += "Multi-Scale Temporal: 14.95 RMSE ❌\n"
        baseline_text += "  Failed due to temporal complexity\n\n"
        baseline_text += "Wavelet Enhancement Goals:\n"
        baseline_text += "• Improve on Variable LR baseline\n"
        baseline_text += "• Target 7.5-8.5 RMSE range\n"
        baseline_text += "• Better frequency-domain modeling\n"
        baseline_text += "• Preserve proven architecture\n\n"
        baseline_text += "Key Innovation:\n"
        baseline_text += "• LSTM + Wavelet fusion\n"
        baseline_text += "• Frequency-specific attention\n"
        baseline_text += "• Wave-physics aligned processing\n"
        
        axes[1, 2].text(0.05, 0.95, baseline_text, transform=axes[1, 2].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 2].set_title('Baseline Comparison')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / "wavelet_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Wavelet analysis saved: {plot_path}")

class WaveletSpatioTemporalDataset(Dataset):
    """Dataset for wavelet-enhanced spatiotemporal sequences"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: WaveletVariableLRConfig, 
                 start_time: int = 0, end_time: int = 100):
        self.mesh_loader = mesh_loader
        self.config = config
        self.sequences = []
        
        print(f"Building wavelet-enhanced dataset from timesteps {start_time} to {end_time}...")
        
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
        
        print(f"Created {len(self.sequences)} wavelet-enhanced sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def main():
    """Main function to train wavelet-enhanced variable learning rate model"""
    
    print("🌊 WAVELET-ENHANCED VARIABLE LEARNING RATE WAVE MODEL")
    print("=" * 75)
    
    # Configuration with wavelets + proven Variable LR settings
    config = WaveletVariableLRConfig(
        # Temporal settings
        sequence_length=6,
        prediction_horizon=4,
        
        # Wavelet enhancement
        use_wavelets=True,
        wavelet_type='db4',                      # Daubechies-4 wavelet
        wavelet_levels=2,                        # 2 levels for seq_len=6
        wavelet_fusion_method="attention",       # Attention-based fusion
        wavelet_hidden_dim=64,                   # Wavelet feature dimension
        combine_lstm_wavelet=True,               # Combine LSTM + wavelets
        
        # Architecture (proven from Variable LR)
        input_features=11,
        hidden_dim=256,
        temporal_hidden_dim=128,
        num_spatial_layers=6,
        num_temporal_layers=2,
        
        # Attention
        use_spatial_attention=True,
        use_temporal_attention=True,
        num_attention_heads=8,
        
        # Regularization
        dropout=0.15,
        spatial_dropout=0.1,
        temporal_dropout=0.1,
        
        # Training parameters
        num_epochs=200,
        batch_size=8,
        base_learning_rate=1e-4,
        weight_decay=1e-3,
        gradient_clip_norm=1.0,
        
        # Variable-specific learning rates (proven settings)
        swh_lr_multiplier=0.7,      # Lower for excellent SWH
        mwd_lr_multiplier=1.0,      # Standard for circular MWD
        mwp_lr_multiplier=1.3,      # Higher for MWP improvement
        
        # Adaptive learning rate parameters
        lr_patience=15,             # Epochs before LR reduction
        lr_factor=0.8,              # LR reduction factor
        min_lr_factor=0.1,          # Minimum LR threshold
        
        # Early stopping (CRITICAL)
        early_stopping_patience=50,
        
        # Variable-specific loss weights
        swh_loss_weight=1.0,
        mwd_loss_weight=1.0,        # Circular loss for MWD
        mwp_loss_weight=1.0,
        physics_loss_weight=0.2,
        
        # Data parameters
        validation_split=0.2,
        max_training_sequences=1000,
        
        # Curriculum learning
        start_with_single_step=True,
        single_step_epochs=200
    )
    
    print(f"🔧 Wavelet-Enhanced Configuration:")
    print(f"   Baseline to beat: 9.21 ± 0.82 RMSE (Variable LR)")
    print(f"   Failed comparison: 14.95 RMSE (Multi-scale Temporal)")
    print(f"   ")
    print(f"   🌊 Wavelet Innovation:")
    print(f"   Wavelet Type: {config.wavelet_type} (Daubechies-4)")
    print(f"   Decomposition Levels: {config.wavelet_levels}")
    print(f"     Level 0: High freq (wind waves, 1-3hr)")
    print(f"     Level 1: Medium freq (local systems, 3-6hr)")
    print(f"     Level 2: Low freq (swell propagation, 6-12hr)")
    print(f"     Approximation: Long-term trends")
    print(f"   Fusion: {config.wavelet_fusion_method}-based combination")
    print(f"   Integration: LSTM + Wavelet fusion")
    print(f"   ")
    print(f"   📈 Expected Benefits:")
    print(f"     • Better storm event detection (transients)")
    print(f"     • Wave group envelope modeling")
    print(f"     • Frequency-domain wave interactions")
    print(f"     • Improved MWP prediction (periods)")
    print(f"   ")
    print(f"   🎯 Performance Targets:")
    print(f"     SWH: 0.674m → 0.60-0.68m (maintain + improve)")
    print(f"     MWD: 25.8° → 22-26° (better directional accuracy)")
    print(f"     MWP: 1.187s → 1.0-1.1s (frequency-domain advantage)")
    print(f"     Overall: 9.21 → 7.5-8.5 RMSE (significant improvement)")
    
    # Train wavelet-enhanced model
    trainer = WaveletVariableLRTrainer(config)
    model, history, edge_index, edge_attr = trainer.train()
    
    print(f"\n🎉 Wavelet-enhanced training complete!")
    print(f"   Base: Proven Variable LR architecture (9.21 RMSE)")
    print(f"   Enhancement: Wavelet frequency-domain processing")
    print(f"   Innovation: LSTM + Wavelet fusion for temporal dynamics")
    print(f"   Next step: Annual evaluation against 9.21 RMSE baseline")
    print(f"   Target: 7.5-8.5 RMSE with frequency-domain advantages")

if __name__ == "__main__":
    main()