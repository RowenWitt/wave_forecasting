#!/usr/bin/env python3
"""
Multi-Scale Temporal Spatiotemporal Wave Model with Variable Learning Rates
Implements multi-scale temporal attention for different wave timescales:
- Wind waves: 1-6 hour generation/decay
- Swell propagation: 12-48 hours
- Storm systems: 2-7 days
- Seasonal patterns: Monthly variations
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
class MultiScaleTemporalConfig:
    """Configuration with multi-scale temporal processing and variable learning rates"""
    
    # Temporal parameters
    sequence_length: int = 6           
    prediction_horizon: int = 4        
    
    # Multi-scale temporal parameters
    use_multiscale_temporal: bool = True
    temporal_scales: List[int] = None  # Will default to [1, 2, 4] (1hr, 2hr, 4hr windows)
    multiscale_fusion_method: str = "attention"  # "attention", "concat", "weighted"
    
    # Model architecture
    input_features: int = 11
    hidden_dim: int = 256
    temporal_hidden_dim: int = 128
    num_spatial_layers: int = 6        
    num_temporal_layers: int = 2       
    
    # Attention
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    num_attention_heads: int = 8
    
    # Regularization
    dropout: float = 0.15
    spatial_dropout: float = 0.1
    temporal_dropout: float = 0.1
    
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 8               
    base_learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Variable-specific learning rates (multipliers)
    swh_lr_multiplier: float = 0.7      # Lower for converged SWH
    mwd_lr_multiplier: float = 1.3      # Standard for circular MWD
    mwp_lr_multiplier: float = 1.3      # Higher for MWP improvement
    
    # Adaptive learning rate parameters
    lr_patience: int = 15               # Epochs to wait before LR adjustment
    lr_factor: float = 0.8              # Factor to reduce LR
    min_lr_factor: float = 0.1          # Minimum LR as fraction of base
    
    # Early stopping (IMPORTANT: Enable this!)
    early_stopping_patience: int = 80  # More aggressive early stopping
    
    # Variable-specific loss weights
    swh_loss_weight: float = 1.0
    mwd_loss_weight: float = 1.0        # Circular loss for MWD
    mwp_loss_weight: float = 1.0
    physics_loss_weight: float = 0.2
    
    # Data
    validation_split: float = 0.2
    max_training_sequences: int = 1000  
    
    # Curriculum learning
    start_with_single_step: bool = True
    single_step_epochs: int = 200
    
    def __post_init__(self):
        """Set default temporal scales if not provided"""
        if self.temporal_scales is None:
            self.temporal_scales = [1, 2, 4]  # 1hr, 2hr, 4hr windows

class VariableLearningManager:
    """Manages variable-specific learning rates and performance tracking"""
    
    def __init__(self, config: MultiScaleTemporalConfig):
        self.config = config
        self.base_lr = config.base_learning_rate
        
        # Current learning rates
        self.current_lrs = {
            'swh': self.base_lr * config.swh_lr_multiplier,
            'mwd': self.base_lr * config.mwd_lr_multiplier,
            'mwp': self.base_lr * config.mwp_lr_multiplier
        }
        
        # Performance tracking
        self.loss_history = {
            'swh': [],
            'mwd': [],
            'mwp': []
        }
        
        # Plateau detection
        self.plateau_counters = {
            'swh': 0,
            'mwd': 0,
            'mwp': 0
        }
        
        self.best_losses = {
            'swh': float('inf'),
            'mwd': float('inf'),
            'mwp': float('inf')
        }
        
        print(f"🎯 Variable Learning Rate Manager initialized:")
        print(f"   SWH LR: {self.current_lrs['swh']:.2e} (converged variable)")
        print(f"   MWD LR: {self.current_lrs['mwd']:.2e} (circular loss)")
        print(f"   MWP LR: {self.current_lrs['mwp']:.2e} (needs improvement)")
    
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
                    self.plateau_counters[var] = 0  # Reset counter
                    print(f"   📉 {var.upper()} LR reduced: {old_lr:.2e} → {new_lr:.2e}")
    
    def get_current_lrs(self) -> Dict[str, float]:
        """Get current learning rates for optimizers"""
        return self.current_lrs.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for logging"""
        summary = {}
        for var in ['swh', 'mwd', 'mwp']:
            if self.loss_history[var]:
                recent_losses = self.loss_history[var][-5:]
                summary[var] = {
                    'current_lr': self.current_lrs[var],
                    'best_loss': self.best_losses[var],
                    'recent_avg': np.mean(recent_losses),
                    'plateau_counter': self.plateau_counters[var]
                }
        return summary

class MultiScaleTemporalProcessor(nn.Module):
    """Multi-scale temporal processing for different wave timescales"""
    
    def __init__(self, hidden_dim: int, temporal_hidden_dim: int, 
                 temporal_scales: List[int], num_temporal_layers: int = 2,
                 fusion_method: str = "attention", use_attention: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_scales = temporal_scales
        self.fusion_method = fusion_method
        self.use_attention = use_attention
        
        print(f"🕐 Multi-Scale Temporal Processor:")
        print(f"   Scales: {temporal_scales} (representing different wave timescales)")
        print(f"   Fusion method: {fusion_method}")
        
        # Create LSTM for each temporal scale
        self.scale_processors = nn.ModuleDict()
        for scale in temporal_scales:
            self.scale_processors[f'scale_{scale}'] = nn.LSTM(
                hidden_dim,
                temporal_hidden_dim // len(temporal_scales),  # Divide hidden dim across scales
                num_temporal_layers,
                batch_first=True,
                dropout=0.1 if num_temporal_layers > 1 else 0
            )
        
        # Temporal attention for each scale
        if use_attention:
            self.scale_attentions = nn.ModuleDict()
            scale_hidden_dim = temporal_hidden_dim // len(temporal_scales)
            # Ensure num_heads divides evenly into scale_hidden_dim
            num_heads_per_scale = max(1, scale_hidden_dim // 16)  # 16 dims per head minimum
            for scale in temporal_scales:
                self.scale_attentions[f'scale_{scale}'] = TemporalAttention(
                    scale_hidden_dim, 
                    num_heads=num_heads_per_scale
                )
        
        # Fusion mechanism
        if fusion_method == "attention":
            # For attention fusion, we need consistent dimensions
            # Each scale outputs temporal_hidden_dim // len(scales), so total is temporal_hidden_dim
            self.scale_fusion_attention = nn.MultiheadAttention(
                temporal_hidden_dim, 
                num_heads=max(1, temporal_hidden_dim // 64),  # Ensure head_dim compatibility
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(temporal_hidden_dim)
            
            # Pre-create fusion projection for attention method
            total_scale_dim = (temporal_hidden_dim // len(temporal_scales)) * len(temporal_scales)
            if total_scale_dim != temporal_hidden_dim:
                self.fusion_projection = nn.Linear(total_scale_dim, temporal_hidden_dim)
            else:
                self.fusion_projection = None
                
        elif fusion_method == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(len(temporal_scales)))
            # Pre-create weighted projection if needed
            scale_dim = temporal_hidden_dim // len(temporal_scales)
            if scale_dim != temporal_hidden_dim:
                self.weighted_projection = nn.Linear(scale_dim, temporal_hidden_dim)
            else:
                self.weighted_projection = None
                
        elif fusion_method == "concat":
            # Output will be concatenated, so each scale gets full hidden_dim
            for scale in temporal_scales:
                self.scale_processors[f'scale_{scale}'] = nn.LSTM(
                    hidden_dim,
                    temporal_hidden_dim,  # Full hidden dim for concat
                    num_temporal_layers,
                    batch_first=True,
                    dropout=0.1 if num_temporal_layers > 1 else 0
                )
            
            self.fusion_projection = nn.Linear(
                temporal_hidden_dim * len(temporal_scales), 
                temporal_hidden_dim
            )
        else:
            # Simple average - may need projection
            scale_dim = temporal_hidden_dim // len(temporal_scales)
            if scale_dim != temporal_hidden_dim:
                self.average_projection = nn.Linear(scale_dim, temporal_hidden_dim)
            else:
                self.average_projection = None
    
    def _create_multiscale_views(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create different temporal scale views of the input"""
        batch_size, seq_len, hidden_dim = x.size()
        scale_views = {}
        
        for scale in self.temporal_scales:
            if scale == 1:
                # Use full sequence for scale 1
                scale_views[f'scale_{scale}'] = x
            else:
                # For larger scales, subsample or pool
                if seq_len >= scale:
                    # Average pooling over scale windows
                    # Reshape to create scale-sized windows
                    pad_size = (scale - (seq_len % scale)) % scale
                    if pad_size > 0:
                        # Pad with last timestep
                        padding = x[:, -1:, :].repeat(1, pad_size, 1)
                        x_padded = torch.cat([x, padding], dim=1)
                    else:
                        x_padded = x
                    
                    # Reshape and pool
                    new_seq_len = x_padded.size(1) // scale
                    x_reshaped = x_padded.view(batch_size, new_seq_len, scale, hidden_dim)
                    scale_views[f'scale_{scale}'] = x_reshaped.mean(dim=2)
                else:
                    # If sequence shorter than scale, use global average
                    scale_views[f'scale_{scale}'] = x.mean(dim=1, keepdim=True)
        
        return scale_views
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multiple temporal scales"""
        # x: [batch_size, seq_len, hidden_dim]
        
        # Create multi-scale views
        scale_views = self._create_multiscale_views(x)
        
        # Process each scale
        scale_outputs = []
        
        for scale in self.temporal_scales:
            scale_key = f'scale_{scale}'
            scale_input = scale_views[scale_key]
            
            # LSTM processing
            lstm_out, _ = self.scale_processors[scale_key](scale_input)
            
            # Temporal attention if enabled
            if self.use_attention and scale_key in self.scale_attentions:
                lstm_out = self.scale_attentions[scale_key](lstm_out)
            
            # Use last timestep
            if lstm_out.size(1) > 0:
                scale_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim//scales]
            else:
                scale_output = lstm_out.squeeze(1)
            
            scale_outputs.append(scale_output)
        
        # Fuse multi-scale representations
        if self.fusion_method == "attention":
            # Stack for attention: [batch_size, num_scales, hidden_dim//scales]
            stacked_outputs = torch.stack(scale_outputs, dim=1)
            batch_size, num_scales, scale_hidden_dim = stacked_outputs.shape
            
            # Simple concatenation followed by projection to target size
            concatenated = stacked_outputs.view(batch_size, -1)  # [batch_size, total_dim]
            
            # Project to target temporal_hidden_dim if needed
            if self.fusion_projection is not None:
                projected = self.fusion_projection(concatenated)  # [batch_size, temporal_hidden_dim]
            else:
                projected = concatenated
            
            # Apply attention to the projected representation
            projected_expanded = projected.unsqueeze(1)  # [batch_size, 1, temporal_hidden_dim]
            fused_output, _ = self.scale_fusion_attention(
                projected_expanded, projected_expanded, projected_expanded
            )
            
            # Extract the single output
            fused_output = fused_output.squeeze(1)  # [batch_size, temporal_hidden_dim]
            fused_output = self.fusion_norm(fused_output)
            
        elif self.fusion_method == "weighted":
            # Weighted combination
            stacked_outputs = torch.stack(scale_outputs, dim=1)  # [batch_size, num_scales, hidden_dim//scales]
            weights = F.softmax(self.scale_weights, dim=0)
            weighted_output = (stacked_outputs * weights.view(1, -1, 1)).sum(dim=1)
            
            # Project to correct size if needed
            if self.weighted_projection is not None:
                fused_output = self.weighted_projection(weighted_output)
            else:
                fused_output = weighted_output
            
        elif self.fusion_method == "concat":
            # Concatenate and project
            concatenated = torch.cat(scale_outputs, dim=-1)  # [batch_size, temporal_hidden_dim * num_scales]
            fused_output = self.fusion_projection(concatenated)
        
        else:
            # Simple average with projection
            stacked_outputs = torch.stack(scale_outputs, dim=1)
            averaged = stacked_outputs.mean(dim=1)
            
            if self.average_projection is not None:
                fused_output = self.average_projection(averaged)
            else:
                fused_output = averaged
        
        return fused_output

class VariableSpecificLoss(nn.Module):
    """Loss function with variable-specific components"""
    
    def __init__(self, config: MultiScaleTemporalConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute variable-specific losses
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
        
        # MWD Loss (circular - both component and angular)
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
        
        # Physics constraints (light)
        physics_loss = self._compute_physics_constraints(predictions)
        
        # Weighted total loss
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
    
    def _compute_physics_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """Light physics constraints"""
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
        
        return swh_penalty + mwp_penalty + 0.1 * circle_penalty

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
    """Multi-head temporal attention with safe dimension handling"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Ensure num_heads divides evenly into hidden_dim
        if hidden_dim % num_heads != 0:
            # Adjust num_heads to be a valid divisor
            valid_heads = []
            for h in range(1, min(num_heads + 1, hidden_dim + 1)):
                if hidden_dim % h == 0:
                    valid_heads.append(h)
            num_heads = max(valid_heads) if valid_heads else 1
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"    TemporalAttention: hidden_dim={hidden_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
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

class MultiScaleTemporalSpatioTemporalGNN(nn.Module):
    """Spatiotemporal model with multi-scale temporal processing and variable learning rates"""
    
    def __init__(self, config: MultiScaleTemporalConfig):
        super().__init__()
        self.config = config
        
        print(f"🌊 Multi-Scale Temporal GNN Architecture:")
        print(f"   Base Hidden Dim: {config.hidden_dim}")
        print(f"   Temporal Scales: {config.temporal_scales}")
        print(f"   Fusion Method: {config.multiscale_fusion_method}")
        
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
        
        # Multi-scale temporal processing (key innovation)
        if config.use_multiscale_temporal:
            self.temporal_processor = MultiScaleTemporalProcessor(
                hidden_dim=config.hidden_dim,
                temporal_hidden_dim=config.temporal_hidden_dim,
                temporal_scales=config.temporal_scales,
                num_temporal_layers=config.num_temporal_layers,
                fusion_method=config.multiscale_fusion_method,
                use_attention=config.use_temporal_attention
            )
        else:
            # Fallback to standard temporal processing
            self.temporal_encoder = nn.LSTM(
                config.hidden_dim,
                config.temporal_hidden_dim,
                config.num_temporal_layers,
                batch_first=True,
                dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
            )
            
            if config.use_temporal_attention:
                self.temporal_attention = TemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Variable-specific prediction heads
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
        """Forward pass with multi-scale temporal processing"""
        
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
        
        # Multi-scale temporal processing for each node
        temporal_outputs = []
        for n in range(num_nodes):
            node_sequence = spatial_sequence[:, :, n, :]  # [batch_size, seq_len, hidden_dim]
            
            if self.config.use_multiscale_temporal:
                # Use multi-scale temporal processor
                final_state = self.temporal_processor(node_sequence)
            else:
                # Standard temporal processing
                lstm_out, _ = self.temporal_encoder(node_sequence)
                
                # Temporal attention
                if self.config.use_temporal_attention:
                    lstm_out = self.temporal_attention(lstm_out)
                
                # Use last timestep
                final_state = lstm_out[:, -1, :]
            
            temporal_outputs.append(final_state)
        
        # Stack node outputs
        temporal_features = torch.stack(temporal_outputs, dim=1)
        
        # Shared feature transformation
        shared_features = self.shared_head(temporal_features)
        
        # Variable-specific predictions
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
        shared_params.extend(self.shared_head.parameters())
        
        # Multi-scale temporal or standard temporal
        if self.config.use_multiscale_temporal:
            shared_params.extend(self.temporal_processor.parameters())
        else:
            shared_params.extend(self.temporal_encoder.parameters())
            if hasattr(self, 'temporal_attention'):
                shared_params.extend(self.temporal_attention.parameters())
        
        if hasattr(self, 'spatial_attention'):
            shared_params.extend(self.spatial_attention.parameters())
        
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

class MultiScaleTemporalTrainer:
    """Trainer with multi-scale temporal processing and variable learning rates"""
    
    def __init__(self, config: MultiScaleTemporalConfig):
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
        
        # Learning rate manager
        self.lr_manager = VariableLearningManager(config)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"multiscale_temporal_varlr_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Multi-Scale Temporal Training: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup data (same as before)"""
        
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
        
        dataset = SpatioTemporalDataset(mesh_loader, self.config, start_time=0, end_time=100)
        
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

    def create_optimizers(self, model: MultiScaleTemporalSpatioTemporalGNN):
        """Create variable-specific optimizers"""
        
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
        """Train one epoch with variable learning rates"""
        
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
        """Validate model"""
        
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
        """Main training loop with multi-scale temporal processing"""
        
        print(f"🚀 Starting multi-scale temporal training...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        
        # Create model and loss
        model = MultiScaleTemporalSpatioTemporalGNN(self.config).to(self.device)
        criterion = VariableSpecificLoss(self.config)
        
        # Create variable-specific optimizers
        optimizers = self.create_optimizers(model)
        
        print(f"✅ Multi-Scale Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Temporal Scales: {self.config.temporal_scales}")
        print(f"   Fusion Method: {self.config.multiscale_fusion_method}")
        print(f"   Variable-specific heads: SWH, MWD (circular), MWP")
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': {},
            'variable_losses': {'swh': [], 'mwd': [], 'mwp': []}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        print(f"\n📈 Multi-scale temporal training ({self.config.single_step_epochs} epochs)")
        
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
                print(f"   📊 Multi-Scale Performance Summary:")
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
                }, self.log_dir / "best_multiscale_temporal_model.pt")
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
        
        model_path = self.log_dir / "multiscale_temporal_model.pt"
        torch.save(model_data, model_path)
        
        # Save analysis
        self._save_analysis(history)
        
        print(f"✅ Multi-scale temporal training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved: {model_path}")
        
        return model, history, edge_index, edge_attr
    
    def _save_analysis(self, history: Dict[str, Any]):
        """Save multi-scale temporal analysis and plots"""
        
        # Create analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
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
        
        # Multi-scale architecture summary
        arch_text = "Multi-Scale Temporal Architecture:\n\n"
        arch_text += f"Temporal Scales: {self.config.temporal_scales}\n"
        arch_text += f"Fusion Method: {self.config.multiscale_fusion_method}\n"
        arch_text += f"Use Temporal Attention: {self.config.use_temporal_attention}\n"
        arch_text += f"Hidden Dim: {self.config.hidden_dim}\n"
        arch_text += f"Temporal Hidden Dim: {self.config.temporal_hidden_dim}\n\n"
        arch_text += "Scale Interpretation:\n"
        for scale in self.config.temporal_scales:
            if scale == 1:
                arch_text += f"  Scale {scale}: Wind waves (1-6hr)\n"
            elif scale == 2:
                arch_text += f"  Scale {scale}: Local systems (2-12hr)\n"
            elif scale == 4:
                arch_text += f"  Scale {scale}: Swell propagation (4-24hr)\n"
            else:
                arch_text += f"  Scale {scale}: Long-term patterns\n"
        
        axes[1, 0].text(0.05, 0.95, arch_text, transform=axes[1, 0].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 0].set_title('Architecture Summary')
        axes[1, 0].axis('off')
        
        # Performance summary
        final_perf = self.lr_manager.get_performance_summary()
        summary_text = "Final Performance Summary:\n\n"
        for var, stats in final_perf.items():
            summary_text += f"{var.upper()}:\n"
            summary_text += f"  Final LR: {stats['current_lr']:.2e}\n"
            summary_text += f"  Best Loss: {stats['best_loss']:.4f}\n"
            summary_text += f"  Recent Avg: {stats['recent_avg']:.4f}\n\n"
        
        summary_text += f"\nTarget Improvements:\n"
        summary_text += f"SWH: 0.683m → 0.65-0.70m\n"
        summary_text += f"MWD: 32.6° → 28-32°\n"
        summary_text += f"MWP: 1.160s → 1.0-1.1s\n"
        summary_text += f"Overall: 11.47 → <9.0 RMSE\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        # Comparison to baseline
        baseline_text = "vs Baseline (Variable LR Model):\n\n"
        baseline_text += "Baseline: 9.21 ± 0.82 RMSE\n"
        baseline_text += "  SWH: 0.674m\n"
        baseline_text += "  MWD: 25.8°\n"
        baseline_text += "  MWP: 1.187s\n\n"
        baseline_text += "Multi-Scale Innovation:\n"
        baseline_text += "• Parallel temporal processing\n"
        baseline_text += "• Wave timescale separation\n"
        baseline_text += "• Adaptive scale fusion\n"
        baseline_text += "• Improved temporal dynamics\n\n"
        baseline_text += "Expected Benefits:\n"
        baseline_text += "• Better storm event capture\n"
        baseline_text += "• Improved swell propagation\n"
        baseline_text += "• Enhanced MWP prediction\n"
        
        axes[1, 2].text(0.05, 0.95, baseline_text, transform=axes[1, 2].transAxes, 
                        fontfamily='monospace', verticalalignment='top', fontsize=9)
        axes[1, 2].set_title('Innovation Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / "multiscale_temporal_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Multi-scale temporal analysis saved: {plot_path}")

class SpatioTemporalDataset(Dataset):
    """Dataset for spatiotemporal sequences"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: MultiScaleTemporalConfig, 
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
        
        print(f"Created {len(self.sequences)} spatiotemporal sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def main():
    """Main function to train multi-scale temporal model"""
    
    print("🌊 MULTI-SCALE TEMPORAL SPATIOTEMPORAL WAVE MODEL")
    print("=" * 70)
    
    # Configuration with multi-scale temporal processing
    config = MultiScaleTemporalConfig(
        # Temporal settings
        sequence_length=6,
        prediction_horizon=4,
        
        # Multi-scale temporal innovation
        use_multiscale_temporal=True,
        temporal_scales=[1, 2, 4],  # 1hr, 2hr, 4hr windows
        multiscale_fusion_method="attention",  # "attention", "concat", "weighted"
        
        # Architecture (proven baseline)
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
        
        # Variable-specific learning rates (proven from baseline)
        swh_lr_multiplier=0.7,      # Lower for excellent SWH (0.674m)
        mwd_lr_multiplier=1.0,      # Standard for good MWD (25.8°)
        mwp_lr_multiplier=1.3,      # Higher for improvable MWP (1.187s)
        
        # Adaptive learning rate parameters
        lr_patience=15,             # Epochs before LR reduction
        lr_factor=0.8,              # LR reduction factor
        min_lr_factor=0.1,          # Minimum LR threshold
        
        # Early stopping (CRITICAL: Prevent overfitting!)
        early_stopping_patience=50,  # More aggressive early stopping,
        
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
    
    print(f"🔧 Multi-Scale Temporal Configuration:")
    print(f"   Baseline to beat: 9.21 ± 0.82 overall RMSE")
    print(f"   Current best: SWH=0.674m, MWD=25.8°, MWP=1.187s")
    print(f"   ")
    print(f"   🕐 Multi-Scale Innovation:")
    print(f"   Temporal Scales: {config.temporal_scales}")
    print(f"     Scale 1: Wind waves (1-6 hour dynamics)")
    print(f"     Scale 2: Local systems (2-12 hour evolution)")  
    print(f"     Scale 4: Swell propagation (4-24 hour patterns)")
    print(f"   Fusion Method: {config.multiscale_fusion_method}")
    print(f"   ")
    print(f"   📈 Expected Multi-Scale Benefits:")
    print(f"     • Better storm event capture")
    print(f"     • Improved swell propagation modeling")
    print(f"     • Enhanced MWP prediction (target improvement)")
    print(f"     • More accurate temporal dynamics")
    print(f"   ")
    print(f"   🎯 Performance Targets:")
    print(f"     SWH: 0.674m → 0.65-0.68m (maintain excellence)")
    print(f"     MWD: 25.8° → 24-26° (slight improvement)")
    print(f"     MWP: 1.187s → 1.05-1.15s (significant target)")
    print(f"     Overall: 9.21 → 8.5-9.0 RMSE (consistent sub-9)")
    
    # Train model with multi-scale temporal processing
    trainer = MultiScaleTemporalTrainer(config)
    model, history, edge_index, edge_attr = trainer.train()
    
    print(f"\n🎉 Multi-scale temporal training complete!")
    print(f"   Model enhanced with parallel temporal scale processing")
    print(f"   Innovation: Captures wind waves, swell propagation, and storm systems")
    print(f"   Next step: Annual evaluation against 9.21 RMSE baseline")
    print(f"   Target: Break into 8.5-9.0 RMSE range with improved temporal dynamics")

if __name__ == "__main__":
    main()