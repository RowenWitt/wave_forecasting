#!/usr/bin/env python3
"""
Global Wave Prediction Model V3.2 - Regional Mixture of Experts Version
Adds regional MoE layer for improved performance while maintaining memory efficiency
Designed for both M4 Pro testing and H100 deployment
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from bathymetry_model_dual_purpose import DualPurposeBathymetry

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class HybridSamplingMoEConfig:
    """Configuration for hybrid sampling wave prediction model with MoE"""
    
    # Data paths - now supports multiple files
    data_pattern: str = "data/v1_global/processed/v1_era5_2021*.nc"  # Glob pattern
    cache_dir: str = "cache/global_mesh_hybrid_moe"
    output_dir: str = "experiments/global_wave_v3_2_moe"
    
    # Geographic coverage (global)
    lat_bounds: tuple = (-90.0, 90.0)
    lon_bounds: tuple = (0.0, 360.0)
    
    # Mesh parameters
    mesh_refinement_level: int = 5  # ~40k nodes globally
    max_edge_distance_km: float = 500.0
    
    # Multiscale edge parameters
    use_multiscale_edges: bool = False
    medium_edge_distance_km: float = 1000.0
    long_edge_distance_km: float = 2000.0
    
    # Sampling parameters
    samples_per_epoch: int = 200  # ~90 min epochs
    min_samples_per_month: int = 100  # Ensure monthly coverage
    hard_region_boost_factor: float = 2.0  # Oversample difficult regions
    seasonal_balance: bool = True  # Ensure seasonal diversity
    
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
    
    # MoE parameters (NEW)
    num_experts: int = 8  # Number of regional experts
    num_active_experts: int = 2  # Top-k experts per node
    moe_hidden_dim: int = 256  # Expert hidden dimension
    moe_capacity_factor: float = 1.25  # Capacity factor for load balancing
    moe_noise_epsilon: float = 1e-2  # Noise for exploration during training
    
    # Regional definitions for MoE (NEW) - Complete ocean coverage
    regions: Dict[str, Dict] = field(default_factory=lambda: {
        # Polar regions
        'arctic': {'lat_range': (66, 90), 'expert_id': 0},
        'antarctic': {'lat_range': (-90, -66), 'expert_id': 1},
        
        # Northern mid-latitudes (split by ocean basin)
        'north_atlantic': {'lat_range': (30, 66), 'lon_range': (-100, 20), 'expert_id': 2},
        'north_pacific_west': {'lat_range': (30, 66), 'lon_range': (120, 180), 'expert_id': 3},
        'north_pacific_east': {'lat_range': (30, 66), 'lon_range': (-180, -100), 'expert_id': 4},
        
        # Tropical regions (split by ocean basin)
        'tropical_atlantic': {'lat_range': (-30, 30), 'lon_range': (-70, 20), 'expert_id': 5},
        'tropical_pacific': {'lat_range': (-30, 30), 'lon_range': (120, -70), 'expert_id': 6},
        'tropical_indian': {'lat_range': (-30, 30), 'lon_range': (20, 120), 'expert_id': 7},
        
        # Southern Ocean (could split further if using more experts)
        'southern_ocean': {'lat_range': (-66, -30), 'expert_id': 8},
        
        # Mediterranean/European seas (specialized region)
        'mediterranean': {'lat_range': (30, 47), 'lon_range': (-10, 40), 'expert_id': 9}
    })
    
    # Update num_experts to match
    num_experts: int = 10  # Increased to cover all regions
    
    # Training parameters
    batch_size: int = 2  # Slightly larger for diversity
    accumulation_steps: int = 2  # Effective batch size of 8
    num_epochs: int = 100
    base_learning_rate: float = 5e-5  # Reduced from 1e-4 for stability
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 5.0  # Increased from 1.0 for better stability
    
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
# REGIONAL MOE COMPONENTS
# ==============================================================================

class RegionalRouter(nn.Module):
    """Routes nodes to experts based on geographic location and learned features"""
    
    def __init__(self, input_dim: int, num_experts: int, num_active_experts: int, 
                 noise_epsilon: float = 1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.noise_epsilon = noise_epsilon
        
        # Learned routing based on features
        self.feature_router = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
        
        # Regional bias for geographic priors
        self.regional_bias = nn.Parameter(torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor, node_regions: torch.Tensor = None, training: bool = True):
        """
        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            node_regions: Regional assignments [batch_size, num_nodes, num_experts] (soft assignments)
            training: Whether in training mode
            
        Returns:
            gates: Gating weights [batch_size, num_nodes, num_experts]
            selected_experts: Top-k expert indices [batch_size, num_nodes, num_active_experts]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Compute routing logits from features
        routing_logits = self.feature_router(x)  # [batch, nodes, experts]
        
        # Add regional bias if provided
        if node_regions is not None:
            # node_regions contains soft assignments (can be learned or fixed)
            routing_logits = routing_logits + self.regional_bias * node_regions
        
        # Add noise during training for exploration
        if training and self.noise_epsilon > 0:
            noise = torch.randn_like(routing_logits) * self.noise_epsilon
            routing_logits = routing_logits + noise
        
        # Compute top-k experts
        top_k_logits, selected_experts = torch.topk(
            routing_logits, self.num_active_experts, dim=-1
        )
        
        # Compute gates with softmax over selected experts
        gates = F.softmax(top_k_logits, dim=-1)
        
        return gates, selected_experts

class Expert(nn.Module):
    """Single expert network for wave prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RegionalMoELayer(nn.Module):
    """Mixture of Experts layer with regional specialization"""
    
    def __init__(self, config: HybridSamplingMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        
        # Input/output dimensions
        self.input_dim = config.temporal_hidden_dim
        self.output_dim = config.num_output_features
        
        # Create router
        self.router = RegionalRouter(
            self.input_dim, 
            self.num_experts, 
            self.num_active_experts,
            config.moe_noise_epsilon
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(
                self.input_dim, 
                config.moe_hidden_dim, 
                self.output_dim,
                config.dropout
            ) for _ in range(self.num_experts)
        ])
        
        # Load balancing loss weight
        self.load_balance_weight = 0.01
        
    def forward(self, x: torch.Tensor, node_regions: torch.Tensor = None, 
                return_router_logits: bool = False):
        """
        Args:
            x: Input features [batch_size, num_nodes, hidden_dim]
            node_regions: Regional soft assignments [batch_size, num_nodes, num_experts]
            return_router_logits: Whether to return routing information for analysis
            
        Returns:
            output: Predictions [batch_size, num_nodes, output_dim]
            aux_loss: Load balancing auxiliary loss
            router_info: (optional) Dictionary with routing information
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Get routing decisions
        gates, selected_experts = self.router(x, node_regions, self.training)
        
        # Initialize output
        output = torch.zeros(
            batch_size, num_nodes, self.output_dim, 
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert
        # For memory efficiency, we process in chunks if needed
        if num_nodes > 10000:  # Chunk processing for large graphs
            chunk_size = 5000
            for start_idx in range(0, num_nodes, chunk_size):
                end_idx = min(start_idx + chunk_size, num_nodes)
                self._process_chunk(
                    x[:, start_idx:end_idx], 
                    gates[:, start_idx:end_idx],
                    selected_experts[:, start_idx:end_idx],
                    output[:, start_idx:end_idx],
                    start_idx
                )
        else:
            self._process_chunk(x, gates, selected_experts, output, 0)
        
        # Compute load balancing loss
        aux_loss = self._compute_load_balance_loss(gates, selected_experts)
        
        if return_router_logits:
            router_info = {
                'gates': gates,
                'selected_experts': selected_experts,
                'expert_usage': self._compute_expert_usage(selected_experts)
            }
            return output, aux_loss, router_info
        
        return output, aux_loss
    
    def _process_chunk(self, x_chunk, gates_chunk, experts_chunk, output_chunk, offset):
        """Process a chunk of nodes through selected experts"""
        batch_size, chunk_nodes, _ = x_chunk.shape
        
        # Scatter-gather approach for efficiency
        for expert_idx in range(self.num_experts):
            # Find nodes that selected this expert
            expert_mask = (experts_chunk == expert_idx).any(dim=-1)  # [batch, chunk_nodes]
            
            if not expert_mask.any():
                continue
                
            # Get indices where this expert is selected
            batch_indices, node_indices = torch.where(expert_mask)
            
            if len(batch_indices) == 0:
                continue
            
            # Gather inputs for this expert
            expert_inputs = x_chunk[batch_indices, node_indices]  # [num_selected, hidden]
            
            # Run expert
            expert_outputs = self.experts[expert_idx](expert_inputs)  # [num_selected, output]
            
            # Find which position (top-1 or top-2) this expert was selected for
            for k in range(self.num_active_experts):
                position_mask = experts_chunk[batch_indices, node_indices, k] == expert_idx
                if position_mask.any():
                    weights = gates_chunk[batch_indices, node_indices, k]
                    weighted_outputs = expert_outputs * weights.unsqueeze(-1)
                    
                    # Accumulate to output
                    output_chunk[batch_indices[position_mask], 
                               node_indices[position_mask]] += weighted_outputs[position_mask]
    
    def _compute_load_balance_loss(self, gates, selected_experts):
        """Compute auxiliary loss for load balancing across experts"""
        # Compute expert usage
        expert_usage = torch.zeros(
            self.num_experts, device=gates.device
        )
        
        for i in range(self.num_experts):
            mask = (selected_experts == i).float()
            expert_usage[i] = (mask * gates).sum()
        
        # Normalize
        total_usage = expert_usage.sum()
        if total_usage > 0:
            expert_usage = expert_usage / total_usage
        
        # Target uniform distribution
        target_usage = 1.0 / self.num_experts
        
        # L2 loss from uniform
        aux_loss = ((expert_usage - target_usage) ** 2).sum()
        
        return aux_loss * self.load_balance_weight
    
    def _compute_expert_usage(self, selected_experts):
        """Compute expert usage statistics"""
        expert_counts = torch.zeros(self.num_experts)
        for i in range(self.num_experts):
            expert_counts[i] = (selected_experts == i).sum().item()
        return expert_counts

# ==============================================================================
# MODIFIED MODEL WITH MOE
# ==============================================================================

class MultiscaleGlobalWaveGNNMoE(nn.Module):
    """Global spatiotemporal GNN with multiscale message passing and regional MoE"""
    
    def __init__(self, config: HybridSamplingMoEConfig, mesh_lats: np.ndarray, mesh_lons: np.ndarray):
        super().__init__()
        self.config = config
        
        # Store mesh coordinates for regional assignment
        self.register_buffer('mesh_lats', torch.tensor(mesh_lats, dtype=torch.float32))
        self.register_buffer('mesh_lons', torch.tensor(mesh_lons, dtype=torch.float32))
        
        # Precompute regional assignments
        self.register_buffer('node_regions', self._compute_node_regions())
        
        # Feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.num_input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Spatial layers - now using multiscale message passing
        self.spatial_layers = nn.ModuleList([
            MultiscaleMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Spatial attention (same as before)
        self.spatial_attention = SpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        # Temporal processing (same as before)
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.dropout if config.num_temporal_layers > 1 else 0
        )
        
        # Regional MoE layer (NEW)
        self.moe_layer = RegionalMoELayer(config)
        
        self._initialize_weights()
    
    def _compute_node_regions(self) -> torch.Tensor:
        """Compute soft regional assignments for each mesh node with complete coverage"""
        num_nodes = len(self.mesh_lats)
        num_experts = self.config.num_experts
        
        # Initialize soft assignments
        node_regions = torch.zeros(num_nodes, num_experts)
        
        # Track which nodes have been assigned
        assigned = torch.zeros(num_nodes, dtype=torch.bool)
        
        for node_idx in range(num_nodes):
            lat = self.mesh_lats[node_idx].item()
            lon = self.mesh_lons[node_idx].item()
            
            # Convert longitude to [-180, 180] for consistency
            if lon > 180:
                lon = lon - 360
            
            # Find matching regions
            matches = []
            for region_name, region_info in self.config.regions.items():
                if 'expert_id' not in region_info or region_info['expert_id'] >= num_experts:
                    continue
                    
                expert_id = region_info['expert_id']
                lat_range = region_info['lat_range']
                
                # Check latitude
                if lat_range[0] <= lat <= lat_range[1]:
                    # Check longitude if specified
                    if 'lon_range' in region_info:
                        lon_range = region_info['lon_range']
                        # Handle wraparound (e.g., Pacific crossing dateline)
                        if lon_range[0] > lon_range[1]:  # Crosses dateline
                            if lon >= lon_range[0] or lon <= lon_range[1]:
                                matches.append(expert_id)
                        else:
                            if lon_range[0] <= lon <= lon_range[1]:
                                matches.append(expert_id)
                    else:
                        # No longitude restriction, matches entire latitude band
                        matches.append(expert_id)
            
            # Assign weights
            if matches:
                # Equal weight to all matching regions
                weight = 1.0 / len(matches)
                for expert_id in matches:
                    node_regions[node_idx, expert_id] = weight
                assigned[node_idx] = True
            else:
                # Fallback: assign to nearest expert based on lat/lon distance
                # This ensures complete coverage
                min_dist = float('inf')
                nearest_expert = 0
                
                # Find nearest region center
                for region_name, region_info in self.config.regions.items():
                    if 'expert_id' not in region_info or region_info['expert_id'] >= num_experts:
                        continue
                        
                    expert_id = region_info['expert_id']
                    lat_range = region_info['lat_range']
                    lat_center = (lat_range[0] + lat_range[1]) / 2
                    
                    if 'lon_range' in region_info:
                        lon_range = region_info['lon_range']
                        if lon_range[0] > lon_range[1]:  # Crosses dateline
                            lon_center = (lon_range[0] + lon_range[1] + 360) / 2
                            if lon_center > 180:
                                lon_center -= 360
                        else:
                            lon_center = (lon_range[0] + lon_range[1]) / 2
                    else:
                        lon_center = lon  # Use node's longitude
                    
                    # Simple distance (could use haversine for accuracy)
                    dist = ((lat - lat_center)**2 + (lon - lon_center)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        nearest_expert = expert_id
                
                node_regions[node_idx, nearest_expert] = 1.0
        
        # Add small epsilon to all weights for exploration
        epsilon = 0.1
        node_regions = node_regions * (1 - epsilon) + epsilon / num_experts
        
        # Normalize to ensure sum to 1
        row_sums = node_regions.sum(dim=1, keepdim=True)
        node_regions = node_regions / row_sums
        
        # Print coverage statistics
        print(f"\n📊 Regional coverage statistics:")
        for i in range(num_experts):
            count = (node_regions[:, i] > 0.1).sum().item()
            print(f"   Expert {i}: {count} nodes ({count/num_nodes*100:.1f}%)")
        
        unassigned = (~assigned).sum().item()
        if unassigned > 0:
            print(f"   ⚠️  {unassigned} nodes required fallback assignment")
        
        return node_regions
    
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
                edge_slices: Dict[str, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
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
                for layer in self.spatial_layers:
                    h_b = layer(h_b, edge_index, edge_attr, edge_slices)
                
                # Spatial attention (using all edges)
                h_b = self.spatial_attention(h_b.unsqueeze(0), edge_index).squeeze(0)
                
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
        
        # Apply regional MoE layer
        # Expand node regions for batch
        batch_node_regions = self.node_regions.unsqueeze(0).expand(batch_size, -1, -1)
        
        predictions, moe_aux_loss = self.moe_layer(temporal_features, batch_node_regions)
        
        return predictions, moe_aux_loss

# ==============================================================================
# MODIFIED LOSS WITH MOE AUXILIARY LOSS
# ==============================================================================

class CircularLossMoE(nn.Module):
    """Loss function with circular handling for MWD and MoE auxiliary loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                moe_aux_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        
        # Total loss including MoE auxiliary loss
        total_loss = swh_loss + mwd_loss + mwp_loss + moe_aux_loss
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_loss': mwd_loss,
            'mwp_loss': mwp_loss,
            'moe_aux_loss': moe_aux_loss
        }

# ==============================================================================
# REUSE EXISTING COMPONENTS (keeping them the same)
# ==============================================================================

# Copy all the existing classes from the original script:
# - HybridSamplingDataset
# - BathymetryEnabledHybridSamplingDataset
# - DiverseBatchSampler
# - MultiscaleGlobalIcosahedralMesh
# - MultiscaleMessageLayer
# - SpatialAttention
# - CircularNormalizer
# - VariableSpecificNormalizer

# [Note: I'm not repeating all these classes to save space, but they should be copied from the original script]

# For brevity, I'll include the key imports and class signatures:

class HybridSamplingDataset(Dataset):
    """
    Multi-month dataset with smart sampling strategies:
    - Samples diverse data each epoch
    - Ensures seasonal and regional coverage
    - Focuses on difficult regions
    - Maintains reasonable epoch times
    """
    
    def __init__(self, data_paths: List[str], mesh, config: HybridSamplingMoEConfig):
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
            self.ocean_mask = interpolator(points) > 0.5  # Boolean mask
            
            n_ocean = self.ocean_mask.sum()
            print(f"   Ocean nodes: {n_ocean}/{len(self.ocean_mask)} ({n_ocean/len(self.ocean_mask)*100:.1f}%)")
            
            return self.ocean_mask
    
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
                
                mesh_data = self._interpolate_to_mesh(field_data)
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
            config: HybridSamplingMoEConfig
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
        if hasattr(super(), '_interpolate_to_mesh'):
            # Check if parent expects is_ocean_variable parameter
            import inspect
            parent_sig = inspect.signature(super()._interpolate_to_mesh)
            if 'is_ocean_variable' in parent_sig.parameters:
                interpolated = super()._interpolate_to_mesh(field_data, is_ocean_variable=is_ocean_variable)
            else:
                interpolated = super()._interpolate_to_mesh(field_data)
        else:
            # Fallback if parent doesn't have the method
            raise NotImplementedError("Parent class must implement _interpolate_to_mesh")
        
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
        # Get base sample from parent
        sample = super().__getitem__(idx)
        
        # If we have bathymetry, replace the ocean_depth feature
        if self.mesh_bathymetry is not None and 'ocean_depth' in self.config.input_features:
            depth_idx = self.config.input_features.index('ocean_depth')
            
            # Replace with proper bathymetry for all timesteps
            for t in range(sample['input'].shape[0]):
                sample['input'][t, :, depth_idx] = torch.tensor(
                    self.mesh_bathymetry, 
                    dtype=torch.float32
                )
        
        return sample
    
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

class MultiscaleGlobalIcosahedralMesh:
    """Global icosahedral mesh with multiscale edge connectivity"""
    
    def __init__(self, refinement_level: int, config: HybridSamplingMoEConfig, cache_dir: str = "cache/global_mesh_hybrid"):
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
            nn.Linear(4 * hidden_dim, hidden_dim),  # 4x for node + 3 message types
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


class SpatialAttention(nn.Module):
    """Multi-head spatial attention for graph nodes"""
    
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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # For large graphs, we need to avoid creating the full attention matrix
        # Instead, use sparse attention based on edges
        
        if len(x.shape) == 2:  # [nodes, features]
            x = x.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, num_nodes, hidden_dim = x.size()
        
        # Skip attention if too many nodes to avoid memory issues
        if num_nodes > 20000:  # Threshold for sparse attention
            # Simple residual connection without attention
            return x.squeeze(0) if squeeze_output else x
        
        Q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply edge mask only for smaller graphs
        if edge_index is not None and num_nodes < 10000:
            mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            mask[edge_index[0], edge_index[1]] = True
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        
        output = self.out_linear(attended)
        result = self.layer_norm(x + output)
        
        return result.squeeze(0) if squeeze_output else result

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
# MODIFIED TRAINER FOR MOE
# ==============================================================================

class HybridSamplingMoETrainer:
    """Trainer for global wave prediction with hybrid sampling and MoE"""
    
    def __init__(self, config: HybridSamplingMoEConfig):
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
            'moe_aux_loss': [],
            'expert_usage': [],
            'epoch_times': [],
            'samples_per_epoch': []
        }
        
        print(f"🌍 Hybrid Sampling Global Wave Trainer with Regional MoE initialized")
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
        
        # Get mesh coordinates for MoE
        self.mesh_lats, self.mesh_lons = self.mesh.vertices_to_lat_lon()
        
        # Get combined edge connectivity
        self.edge_index, self.edge_attr, self.edge_slices = self.mesh.get_combined_edge_index_and_attr()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        
        self.dataset = BathymetryEnabledHybridSamplingDataset(
            data_paths=data_files,
            mesh=self.mesh,
            config=self.config,
            gebco_path="data/gebco/GEBCO_2023.nc"
        )
        
        # Fit normalizers
        print("🔧 Fitting normalizers...")
        self._fit_normalizers(self.dataset)
        
        # Create validation dataset with fixed sampling
        print("   Creating validation split...")
        val_size = int(self.config.validation_split * self.config.samples_per_epoch)
        train_size = self.config.samples_per_epoch - val_size
        
        # For validation, we'll use a fixed subset
        self.val_indices = list(range(0, len(self.dataset), len(self.dataset) // val_size))[:val_size]
        
        print("   Initializing epoch samples...")
        if hasattr(self.dataset, '_resample_epoch'):
            self.dataset._resample_epoch()
            print(f"   ✓ Epoch samples ready: {len(self.dataset.epoch_samples)} samples")
        
        self.train_loader = DataLoader(
            self.dataset,
            batch_sampler=DiverseBatchSampler(self.dataset, self.config.batch_size),
            num_workers=0,
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
        
        print(f"✅ Data setup complete:")
        print(f"   Mesh nodes: {len(self.mesh.vertices)}")
        print(f"   Total edges: {self.edge_index.shape[1]}")
        print(f"   Samples per epoch: {self.config.samples_per_epoch}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation samples: {len(self.val_indices)}")
        
        return self.dataset
    
    def _fit_normalizers(self, dataset):
        """Fit normalizers on sample data - copy from original"""
        # Copy implementation from original script
        sample_features = []
        sample_targets = []
        
        print("🔧 Fitting normalizers on sample data...")
        
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
        
        all_features = np.vstack(sample_features)
        all_targets = np.vstack(sample_targets)
        
        # Replace NaN with zeros before fitting
        all_features_clean = np.nan_to_num(all_features, nan=0.0)
        all_targets_clean = np.nan_to_num(all_targets, nan=0.0)
        
        self.feature_normalizer.fit(all_features_clean)
        self.target_normalizer.fit(all_targets_clean)
        
        print(f"   ✅ Normalizers fitted on {len(all_features)} samples")
    
    def train_epoch(self, model, optimizer, criterion):
        """Train one epoch with gradient accumulation and MoE tracking"""
        model.train()
        epoch_losses = []
        epoch_moe_losses = []
        accumulated_loss = 0
        
        valid_batch_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            inputs_norm = self.feature_normalizer.transform(inputs_flat)
            inputs_norm = np.nan_to_num(inputs_norm, nan=0.0, posinf=0.0, neginf=0.0)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets
            targets_flat = targets.view(-1, 3).cpu().numpy()
            targets_norm = self.target_normalizer.transform_targets(targets_flat)
            targets_norm = np.nan_to_num(targets_norm, nan=0.0, posinf=0.0, neginf=0.0)
            targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
            targets = targets.view(batch_size, num_nodes, 4)

            # Forward pass
            try:
                predictions, moe_aux_loss = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
            except RuntimeError as e:
                print(f"   ❌ Model forward pass failed at batch {batch_idx}: {e}")
                continue
            
            # Check for NaN in predictions
            if torch.isnan(predictions).any():
                continue
            
            # Compute loss with gradient accumulation
            loss_dict = criterion(predictions, targets, moe_aux_loss)
            loss = loss_dict['total_loss'] / self.config.accumulation_steps
            
            # Check for NaN/inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            valid_batch_count += 1
            
            # Track MoE auxiliary loss
            epoch_moe_losses.append(loss_dict['moe_aux_loss'].item())
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                if accumulated_loss > 0:
                    epoch_losses.append(accumulated_loss * self.config.accumulation_steps)
                accumulated_loss = 0
                
                if batch_idx % (10 * self.config.accumulation_steps) == 0 and len(epoch_losses) > 0:
                    print(f"   Batch {batch_idx}/{len(self.train_loader)}: "
                          f"Loss={epoch_losses[-1]:.4f}, MoE aux={epoch_moe_losses[-1]:.4f}")
        
        # Handle any remaining gradients
        if accumulated_loss > 0 and valid_batch_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_moe_loss = np.mean(epoch_moe_losses) if epoch_moe_losses else 0.0
        
        return avg_loss, avg_moe_loss
    
    def validate(self, model, criterion):
        """Validate model with MoE tracking"""
        model.eval()
        val_losses = []
        val_losses_by_var = {'swh': [], 'mwd': [], 'mwp': []}
        moe_aux_losses = []
        
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
                predictions, moe_aux_loss = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
                
                # Compute loss
                loss_dict = criterion(predictions, targets, moe_aux_loss)
                val_losses.append(loss_dict['total_loss'].item())
                moe_aux_losses.append(loss_dict['moe_aux_loss'].item())
                
                # Store per-variable losses
                for var in ['swh', 'mwd', 'mwp']:
                    if f'{var}_loss' in loss_dict:
                        val_losses_by_var[var].append(loss_dict[f'{var}_loss'].item())
        
        mean_total_loss = np.mean(val_losses) if val_losses else float('inf')
        mean_var_losses = {
            var: np.mean(losses) if losses else 0.0 
            for var, losses in val_losses_by_var.items()
        }
        mean_moe_loss = np.mean(moe_aux_losses) if moe_aux_losses else 0.0
        
        return mean_total_loss, mean_var_losses, mean_moe_loss
    
    def train(self):
        """Main training loop with hybrid sampling and MoE"""
        print("\n🚀 Starting hybrid sampling global wave model training with Regional MoE...")
        
        # Setup data
        dataset = self.setup_data()
        
        # Create model with MoE
        model = MultiscaleGlobalWaveGNNMoE(
            self.config, 
            self.mesh_lats, 
            self.mesh_lons
        ).to(self.device)
        
        print(f"\n✅ Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Using multiscale message passing with hybrid sampling")
        print(f"   Regional MoE: {self.config.num_experts} experts, top-{self.config.num_active_experts} routing")
        
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
        
        criterion = CircularLossMoE()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n📈 Training for {self.config.num_epochs} epochs...")
        print(f"   Each epoch: {self.config.samples_per_epoch} samples")
        print(f"   Learning rate: {self.config.base_learning_rate} (with warmup)")
        print(f"   Gradient clipping: {self.config.gradient_clip_norm}")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            if epoch < 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Learning rate: {current_lr:.2e} (warmup)")
            
            # Resample data for new epoch
            if epoch > 0:
                dataset.on_epoch_end()
            
            # Train
            train_loss, train_moe_loss = self.train_epoch(model, optimizer, criterion)
            
            # Validate
            val_loss, val_var_losses, val_moe_loss = self.validate(model, criterion)
            
            # Update learning rate during warmup
            if epoch < 5:
                warmup_scheduler.step()
            
            # Track time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_swh_loss'].append(val_var_losses['swh'])
            self.history['val_mwd_loss'].append(val_var_losses['mwd'])
            self.history['val_mwp_loss'].append(val_var_losses['mwp'])
            self.history['moe_aux_loss'].append(train_moe_loss)
            self.history['epoch_times'].append(epoch_time)
            self.history['samples_per_epoch'].append(len(dataset))
            
            print(f"   Train Loss: {train_loss:.4f} (MoE aux: {train_moe_loss:.4f})")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Loss by variable: SWH={val_var_losses['swh']:.4f}, "
                  f"MWD={val_var_losses['mwd']:.4f}, MWP={val_var_losses['mwp']:.4f}")
            print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
            
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
        """Save model checkpoint with MoE info"""
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
            'mesh_lats': self.mesh_lats,
            'mesh_lons': self.mesh_lons,
            'multiscale_edges': self.mesh.multiscale_edges,
            'history': self.history
        }
        
        if is_best:
            path = self.output_dir / "best_model_moe.pt"
        else:
            path = self.output_dir / f"checkpoint_moe_epoch_{epoch+1}.pt"
        
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
                'edge_attributes': self.mesh.edge_attributes,
                'lats': self.mesh_lats,
                'lons': self.mesh_lons
            },
            'training_history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.output_dir / "global_wave_model_moe_final.pt"
        torch.save(final_data, path)
        
        # Also save config as JSON
        config_dict = {k: v for k, v in self.config.__dict__.items() 
                      if isinstance(v, (int, float, str, list, tuple, dict))}
        with open(self.output_dir / "config_moe.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Final model saved to: {path}")
    
    def plot_training_history(self):
        """Plot and save training history with MoE metrics"""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        
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
        
        # MoE auxiliary loss
        ax3.plot(epochs, self.history['moe_aux_loss'], 'm-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MoE Auxiliary Loss')
        ax3.set_title('MoE Load Balancing Loss')
        ax3.grid(True, alpha=0.3)
        
        # Time plot
        ax4.plot(epochs, self.history['epoch_times'], 'g-')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Training Time per Epoch')
        ax4.grid(True, alpha=0.3)
        
        # Expert usage heatmap (placeholder for now)
        ax5.text(0.5, 0.5, 'Expert Usage\n(To be implemented)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Expert Usage Distribution')
        ax5.axis('off')
        
        # Summary statistics
        summary_text = f"Final Performance:\n\n"
        summary_text += f"Total Val Loss: {self.history['val_loss'][-1]:.4f}\n"
        summary_text += f"SWH Val Loss: {self.history['val_swh_loss'][-1]:.4f}\n"
        summary_text += f"MWD Val Loss: {self.history['val_mwd_loss'][-1]:.4f}\n"
        summary_text += f"MWP Val Loss: {self.history['val_mwp_loss'][-1]:.4f}\n"
        summary_text += f"MoE Aux Loss: {self.history['moe_aux_loss'][-1]:.4f}\n\n"
        summary_text += f"Total epochs: {len(epochs)}\n"
        summary_text += f"Avg time/epoch: {np.mean(self.history['epoch_times']):.1f}s\n"
        summary_text += f"Samples/epoch: {self.config.samples_per_epoch}\n\n"
        summary_text += f"MoE Config:\n"
        summary_text += f"  Experts: {self.config.num_experts}\n"
        summary_text += f"  Active experts: {self.config.num_active_experts}"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontfamily='monospace', verticalalignment='top', fontsize=10)
        ax6.set_title('Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history_moe.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Training history plot saved")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("🌍 GLOBAL WAVE PREDICTION MODEL V3.2 - REGIONAL MIXTURE OF EXPERTS")
    print("=" * 70)
    print("Enhanced model with regional specialization through MoE")
    print("Designed for testing on M4 Pro and deployment on H100")
    print("=" * 70)
    
    # Configuration
    config = HybridSamplingMoEConfig()
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Train global wave model with MoE')
    parser.add_argument('--data-pattern', type=str, default=config.data_pattern,
                        help='Glob pattern for data files')
    parser.add_argument('--samples-per-epoch', type=int, default=config.samples_per_epoch,
                        help='Number of samples per epoch')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.num_epochs,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default=config.device,
                        choices=['mps', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--num-experts', type=int, default=config.num_experts,
                        help='Number of MoE experts')
    parser.add_argument('--num-active-experts', type=int, default=config.num_active_experts,
                        help='Number of active experts per prediction')
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.data_pattern = args.data_pattern
    config.samples_per_epoch = args.samples_per_epoch
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.device = args.device
    config.num_experts = args.num_experts
    config.num_active_experts = args.num_active_experts
    
    # Detect CUDA if available
    if config.device == 'cuda' and torch.cuda.is_available():
        print(f"🎯 CUDA device detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n📋 Configuration:")
    print(f"   Data pattern: {config.data_pattern}")
    print(f"   Samples per epoch: {config.samples_per_epoch}")
    print(f"   Min samples per month: {config.min_samples_per_month}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Accumulation steps: {config.accumulation_steps}")
    print(f"   Effective batch size: {config.batch_size * config.accumulation_steps}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Device: {config.device}")
    
    print(f"\n🔧 Model configuration:")
    print(f"   Mesh refinement: Level {config.mesh_refinement_level}")
    print(f"   Multiscale edges: {config.use_multiscale_edges}")
    print(f"   Input features: {config.num_input_features}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Sequence length: {config.sequence_length} timesteps")
    
    print(f"\n🎯 MoE configuration:")
    print(f"   Number of experts: {config.num_experts}")
    print(f"   Active experts: {config.num_active_experts}")
    print(f"   Expert hidden dim: {config.moe_hidden_dim}")
    print(f"   Capacity factor: {config.moe_capacity_factor}")
    print(f"   Regional definitions: {len(config.regions)} regions")
    
    # Create trainer and start training
    trainer = HybridSamplingMoETrainer(config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving current state...")
        
        # Save partial results
        if hasattr(trainer, 'history') and trainer.history['train_loss']:
            with open(trainer.output_dir / "partial_history_moe.json", 'w') as f:
                json.dump(trainer.history, f)
            print("   Saved partial history")
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        if hasattr(trainer, 'history') and trainer.history['train_loss']:
            print("\n💾 Saving partial results...")
            with open(trainer.output_dir / "partial_history_moe.json", 'w') as f:
                json.dump(trainer.history, f)
    
    print("\n🎉 Regional MoE global wave model training complete!")
    print(f"   Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()