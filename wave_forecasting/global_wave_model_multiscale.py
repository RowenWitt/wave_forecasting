#!/usr/bin/env python3
"""
Global Wave Prediction Model V1 - Multiscale Mesh Version
Single-file implementation with multiscale edge connectivity
Based on proven North Pacific architecture, scaled to global coverage
Optimized for Apple M4 Pro with 128GB RAM
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

# Set memory limit for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable upper limit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class GlobalWaveConfig:
    """Configuration for global wave prediction model"""
    
    # Data paths
    data_path: str = "data/processed_v1/enhanced_v1_era5_202101.nc"
    cache_dir: str = "cache/global_mesh_multiscale"  # Different cache dir for multiscale
    output_dir: str = "experiments/global_wave_v1_multiscale"
    
    # Geographic coverage (global)
    lat_bounds: tuple = (-90.0, 90.0)
    lon_bounds: tuple = (0.0, 360.0)
    
    # Mesh parameters
    mesh_refinement_level: int = 6  # ~40k nodes globally
    max_edge_distance_km: float = 500.0  # For local edges
    
    # Multiscale edge parameters
    use_multiscale_edges: bool = True
    medium_edge_distance_km: float = 1000.0  # Medium-range connections
    long_edge_distance_km: float = 2000.0    # Long-range connections
    
    # Input features (excluding climate anomalies for now)
    input_features: List[str] = field(default_factory=lambda: [
        'tp', 'u10', 'v10', 'msl', 'sst',  # Surface atmospheric (5)
        'swh', 'mwd', 'mwp', 'shww',       # Wave variables (4)
        'u_850', 'u_500', 'v_850', 'v_500', 'z_850', 'z_500',  # Pressure levels (6)
        'ocean_depth'  # Bathymetry (1)
    ])
    num_input_features: int = 16  # Total without anomalies
    
    # Target variables
    target_features: List[str] = field(default_factory=lambda: ['swh', 'mwd', 'mwp'])
    num_output_features: int = 4  # SWH + MWD(cos,sin) + MWP
    
    # Temporal parameters
    sequence_length: int = 4  # 24 hours of input (reduced from 6)
    prediction_horizon: int = 4  # 24 hours ahead
    
    # Model architecture (proven from North Pacific)
    hidden_dim: int = 128  # Reduced from 256
    temporal_hidden_dim: int = 64  # Reduced from 128
    num_spatial_layers: int = 4  # Reduced from 6
    num_temporal_layers: int = 2
    num_attention_heads: int = 4  # Reduced from 8
    dropout: float = 0.15
    
    # Training parameters
    batch_size: int = 2  # Reduced from 4
    num_epochs: int = 100
    base_learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Variable-specific learning rates
    swh_lr_multiplier: float = 0.7
    mwd_lr_multiplier: float = 1.0
    mwp_lr_multiplier: float = 1.3
    
    # Early stopping
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    use_cpu_fallback: bool = True  # Fallback to CPU if MPS fails

# ==============================================================================
# MULTISCALE ICOSAHEDRAL MESH
# ==============================================================================

class MultiscaleGlobalIcosahedralMesh:
    """Global icosahedral mesh with multiscale edge connectivity"""
    
    def __init__(self, refinement_level: int, config: GlobalWaveConfig, cache_dir: str = "cache/global_mesh_multiscale"):
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
            print(f"   Local edges: {len(self.multiscale_edges['local'])}")
            print(f"   Medium edges: {len(self.multiscale_edges['medium'])}")
            print(f"   Long edges: {len(self.multiscale_edges['long'])}")
        else:
            print(f"🔨 Building multiscale global icosahedral mesh (level {self.refinement_level})...")
            print(f"   This will take 10-30 minutes for multiscale connectivity...")
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
            print(f"   Cache size: {cache_file.stat().st_size / (1024**2):.1f} MB")
    
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
# ENHANCED MESSAGE LAYER FOR MULTISCALE
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

# ==============================================================================
# MODEL WITH MULTISCALE SUPPORT
# ==============================================================================

class MultiscaleGlobalWaveGNN(nn.Module):
    """Global spatiotemporal GNN with multiscale message passing"""
    
    def __init__(self, config: GlobalWaveConfig):
        super().__init__()
        self.config = config
        
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
        
        # Output heads (same as before)
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
        
        # Generate predictions
        predictions = self.output_mlp(temporal_features)  # [batch, nodes, 4]
        
        return predictions

# ==============================================================================
# REUSE COMPONENTS FROM ORIGINAL (unchanged)
# ==============================================================================

# Import all the unchanged components
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
# DATA LOADING WITH MULTISCALE MESH
# ==============================================================================

class GlobalWaveDataset(Dataset):
    """Dataset for global wave prediction"""
    
    def __init__(self, data_path: str, mesh: MultiscaleGlobalIcosahedralMesh, config: GlobalWaveConfig,
                 start_idx: int = 0, end_idx: int = None):
        self.data_path = data_path
        self.mesh = mesh
        self.config = config
        
        # Load data
        print(f"📊 Loading global wave data from: {data_path}")
        self.data = xr.open_dataset(data_path)
        
        # Get mesh coordinates
        self.mesh_lats, self.mesh_lons = mesh.vertices_to_lat_lon()
        
        # Time handling
        time_dim = 'time' if 'time' in self.data.dims else 'valid_time'
        self.time_steps = len(self.data[time_dim])
        
        # Sequence indices
        self.start_idx = start_idx
        self.end_idx = end_idx or (self.time_steps - config.sequence_length - config.prediction_horizon)
        
        print(f"✅ Dataset initialized: {self.end_idx - self.start_idx} sequences")
    
    def _interpolate_to_mesh(self, field_data: np.ndarray) -> np.ndarray:
        """Interpolate regular grid data to icosahedral mesh points"""
        from scipy.interpolate import RegularGridInterpolator
        
        # Get grid coordinates
        lats = self.data.latitude.values
        lons = self.data.longitude.values
        
        # Check for NaN values in input
        nan_ratio = np.isnan(field_data).sum() / field_data.size
        if nan_ratio > 0.5:
            print(f"⚠️  Warning: {nan_ratio:.1%} NaN values in field data")
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (lats, lons), field_data,
            method='linear', bounds_error=False, fill_value=0.0  # Use 0 instead of NaN
        )
        
        # Interpolate to mesh points
        points = np.column_stack([self.mesh_lats, self.mesh_lons])
        interpolated = interpolator(points)
        
        # Replace any remaining NaNs with 0
        interpolated = np.nan_to_num(interpolated, nan=0.0)
        
        return interpolated
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # Get time indices
        input_slice = slice(actual_idx, actual_idx + self.config.sequence_length)
        target_idx = actual_idx + self.config.sequence_length
        
        # Extract input features
        input_features = []
        
        for t in range(self.config.sequence_length):
            t_idx = actual_idx + t
            timestep_features = []
            
            for feat in self.config.input_features:
                if feat in self.data.variables:
                    # Get data for this timestep
                    if 'time' in self.data[feat].dims:
                        field_data = self.data[feat].isel(time=t_idx).values
                    elif 'valid_time' in self.data[feat].dims:
                        field_data = self.data[feat].isel(valid_time=t_idx).values
                    else:
                        # Static field (e.g., ocean_depth)
                        field_data = self.data[feat].values
                    
                    # Interpolate to mesh
                    mesh_data = self._interpolate_to_mesh(field_data)
                    timestep_features.append(mesh_data.astype(np.float32))  # Ensure float32
                else:
                    # Feature not found, use zeros
                    timestep_features.append(np.zeros(len(self.mesh.vertices), dtype=np.float32))
            
            input_features.append(np.stack(timestep_features, axis=-1))
        
        # Stack temporal sequence
        inputs = np.stack(input_features, axis=0)  # [seq_len, num_nodes, num_features]
        
        # Extract targets (single timestep)
        target_features = []
        for feat in self.config.target_features:
            if feat in self.data.variables:
                if 'time' in self.data[feat].dims:
                    field_data = self.data[feat].isel(time=target_idx).values
                elif 'valid_time' in self.data[feat].dims:
                    field_data = self.data[feat].isel(valid_time=target_idx).values
                else:
                    field_data = self.data[feat].values
                
                mesh_data = self._interpolate_to_mesh(field_data)
                target_features.append(mesh_data.astype(np.float32))  # Ensure float32
        
        targets = np.stack(target_features, axis=-1)  # [num_nodes, num_targets]
        
        return {
            'input': torch.FloatTensor(inputs),
            'target': torch.FloatTensor(targets),
            'single_step_target': torch.FloatTensor(targets)  # For compatibility
        }

# ==============================================================================
# TRAINING WITH MULTISCALE
# ==============================================================================

class MultiscaleGlobalWaveTrainer:
    """Trainer for global wave prediction model with multiscale edges"""
    
    def __init__(self, config: GlobalWaveConfig):
        self.config = config
        
        # Try to use configured device, fall back to CPU if needed
        try:
            self.device = torch.device(config.device)
            # Test if device works
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
        self.feature_normalizer = StandardScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_swh_loss': [],
            'val_mwd_loss': [],
            'val_mwp_loss': [],
            'epoch_times': []
        }
        
        print(f"🌍 Multiscale Global Wave Trainer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup global dataset and multiscale mesh"""
        print("\n📊 Setting up global dataset with multiscale mesh...")
        
        # Create multiscale global mesh
        self.mesh = MultiscaleGlobalIcosahedralMesh(
            refinement_level=self.config.mesh_refinement_level,
            config=self.config,
            cache_dir=self.config.cache_dir
        )
        
        # Get combined edge connectivity
        self.edge_index, self.edge_attr, self.edge_slices = self.mesh.get_combined_edge_index_and_attr()
        
        # Move to device
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        
        # Create dataset
        dataset = GlobalWaveDataset(
            data_path=self.config.data_path,
            mesh=self.mesh,
            config=self.config
        )
        
        # Fit normalizers on a sample
        print("🔧 Fitting normalizers...")
        self._fit_normalizers(dataset)
        
        # Split dataset
        val_size = int(self.config.validation_split * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Data setup complete:")
        print(f"   Mesh nodes: {len(self.mesh.vertices)}")
        print(f"   Total edges: {self.edge_index.shape[1]}")
        print(f"   - Local edges: {len(self.mesh.multiscale_edges['local'])}")
        print(f"   - Medium edges: {len(self.mesh.multiscale_edges['medium'])}")
        print(f"   - Long edges: {len(self.mesh.multiscale_edges['long'])}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _fit_normalizers(self, dataset):
        """Fit normalizers on sample data"""
        print("🔧 Fitting normalizers...")
        
        # Sample a few batches
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(50, len(dataset)), 5):
            sample = dataset[i]
            features = sample['input'].numpy()
            targets = sample['target'].numpy()
            
            # Flatten features
            features_flat = features.reshape(-1, features.shape[-1])
            sample_features.append(features_flat)
            sample_targets.append(targets)
        
        # Fit normalizers
        all_features = np.vstack(sample_features)
        all_targets = np.vstack(sample_targets)
        
        # Check data quality before fitting
        print(f"   Feature data shape: {all_features.shape}")
        print(f"   Target data shape: {all_targets.shape}")
        
        # Check for NaN values
        feature_nan_ratio = np.isnan(all_features).sum() / all_features.size
        target_nan_ratio = np.isnan(all_targets).sum() / all_targets.size
        
        print(f"   Feature NaN ratio: {feature_nan_ratio:.1%}")
        print(f"   Target NaN ratio: {target_nan_ratio:.1%}")
        
        if feature_nan_ratio > 0.5 or target_nan_ratio > 0.5:
            print("⚠️  High NaN ratio detected! Data quality issues likely.")
        
        # Print statistics for each feature
        print(f"\n   Feature statistics:")
        for i, feat_name in enumerate(self.config.input_features):
            feat_data = all_features[:, i]
            valid_data = feat_data[~np.isnan(feat_data)]
            if len(valid_data) > 0:
                print(f"     {feat_name}: min={valid_data.min():.3f}, max={valid_data.max():.3f}, "
                      f"mean={valid_data.mean():.3f}, NaN%={np.isnan(feat_data).mean()*100:.1f}%")
            else:
                print(f"     {feat_name}: All NaN!")
        
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
            
            # Debug: Check for NaN in raw data
            if torch.isnan(inputs).any():
                print(f"⚠️  NaN detected in inputs at batch {batch_idx}")
                continue
            if torch.isnan(targets).any():
                print(f"⚠️  NaN detected in targets at batch {batch_idx}")
                continue
            
            # Normalize inputs
            batch_size, seq_len, num_nodes, num_features = inputs.size()
            inputs_flat = inputs.view(-1, num_features).cpu().numpy()
            
            # Check for NaN after flattening
            if np.isnan(inputs_flat).any():
                print(f"⚠️  NaN in flattened inputs at batch {batch_idx}")
                # Skip this batch
                continue
            
            inputs_norm = self.feature_normalizer.transform(inputs_flat)
            
            # Check for NaN after normalization
            if np.isnan(inputs_norm).any():
                print(f"⚠️  NaN after feature normalization at batch {batch_idx}")
                # Replace NaN with 0
                inputs_norm = np.nan_to_num(inputs_norm, nan=0.0)
            
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
            
            # Normalize targets
            targets_flat = targets.view(-1, 3).cpu().numpy()
            
            # Check for NaN in targets
            if np.isnan(targets_flat).any():
                print(f"⚠️  NaN in target data at batch {batch_idx}")
                # Skip this batch
                continue
                
            targets_norm = self.target_normalizer.transform_targets(targets_flat)
            
            # Check for NaN after target normalization
            if np.isnan(targets_norm).any():
                print(f"⚠️  NaN after target normalization at batch {batch_idx}")
                targets_norm = np.nan_to_num(targets_norm, nan=0.0)
            
            targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
            targets = targets.view(batch_size, num_nodes, 4)
            
            # Forward pass with multiscale edges
            optimizer.zero_grad()
            predictions = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
            
            # Check for NaN in predictions
            if torch.isnan(predictions).any():
                print(f"⚠️  NaN in model predictions at batch {batch_idx}")
                print(f"   Input stats: min={inputs.min():.3f}, max={inputs.max():.3f}, mean={inputs.mean():.3f}")
                print(f"   Target stats: min={targets.min():.3f}, max={targets.max():.3f}, mean={targets.mean():.3f}")
                continue
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️  NaN loss at batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️  NaN gradient in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                optimizer.zero_grad()  # Clear the bad gradients
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}")
        
        # Return mean loss, or inf if no valid batches
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate(self, model, criterion):
        """Validate model"""
        model.eval()
        val_losses = []
        val_losses_by_var = {'swh': [], 'mwd': [], 'mwp': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
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
                
                # Forward pass with multiscale edges
                predictions = model(inputs, self.edge_index, self.edge_attr, self.edge_slices)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                val_losses.append(loss_dict['total_loss'].item())
                
                # Store per-variable losses
                if 'swh_loss' in loss_dict:
                    val_losses_by_var['swh'].append(loss_dict['swh_loss'].item())
                if 'mwd_loss' in loss_dict:
                    val_losses_by_var['mwd'].append(loss_dict['mwd_loss'].item())
                if 'mwp_loss' in loss_dict:
                    val_losses_by_var['mwp'].append(loss_dict['mwp_loss'].item())
        
        # Calculate mean losses
        mean_total_loss = np.mean(val_losses)
        mean_var_losses = {
            var: np.mean(losses) if losses else 0.0 
            for var, losses in val_losses_by_var.items()
        }
        
        return mean_total_loss, mean_var_losses
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting multiscale global wave model training...")
        
        # Setup data
        train_dataset, val_dataset = self.setup_data()
        
        # Create model
        model = MultiscaleGlobalWaveGNN(self.config).to(self.device)
        print(f"\n✅ Multiscale model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Using multiscale message passing")
        
        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.base_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = CircularLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n📈 Training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion)
            
            # Validate
            val_loss, val_var_losses = self.validate(model, criterion)
            
            # Track time
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_swh_loss'].append(val_var_losses['swh'])
            self.history['val_mwd_loss'].append(val_var_losses['mwd'])
            self.history['val_mwp_loss'].append(val_var_losses['mwp'])
            self.history['epoch_times'].append(epoch_time)
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val Loss by variable: SWH={val_var_losses['swh']:.4f}, "
                  f"MWD={val_var_losses['mwd']:.4f}, MWP={val_var_losses['mwp']:.4f}")
            print(f"   Time: {epoch_time:.1f}s")
            
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
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
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
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.output_dir / "global_wave_model_multiscale_final.pt"
        torch.save(final_data, path)
        
        # Also save config as JSON
        config_dict = {k: v for k, v in self.config.__dict__.items() if isinstance(v, (int, float, str, list, tuple))}
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Final model saved to: {path}")
    
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
        summary_text = f"Final Performance:\n\n"
        summary_text += f"Total Val Loss: {self.history['val_loss'][-1]:.4f}\n"
        summary_text += f"SWH Val Loss: {self.history['val_swh_loss'][-1]:.4f}\n"
        summary_text += f"MWD Val Loss: {self.history['val_mwd_loss'][-1]:.4f}\n"
        summary_text += f"MWP Val Loss: {self.history['val_mwp_loss'][-1]:.4f}\n\n"
        summary_text += f"Total epochs: {len(epochs)}\n"
        summary_text += f"Avg time/epoch: {np.mean(self.history['epoch_times']):.1f}s"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontfamily='monospace', verticalalignment='top', fontsize=12)
        ax4.set_title('Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Training history plot saved")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("🌍 GLOBAL WAVE PREDICTION MODEL V1 - MULTISCALE VERSION")
    print("=" * 70)
    print("Based on proven North Pacific architecture")
    print("Scaled to global coverage with multiscale edge connectivity")
    print("Optimized for Apple M4 Pro with 128GB RAM")
    print("=" * 70)
    
    # Configuration
    config = GlobalWaveConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {config.data_path}")
    print(f"   Mesh refinement: Level {config.mesh_refinement_level}")
    print(f"   Multiscale edges: {config.use_multiscale_edges}")
    print(f"   - Local: < {config.max_edge_distance_km} km")
    print(f"   - Medium: < {config.medium_edge_distance_km} km")
    print(f"   - Long: < {config.long_edge_distance_km} km")
    print(f"   Input features: {config.num_input_features}")
    print(f"   Sequence length: {config.sequence_length} timesteps")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Device: {config.device}")
    
    # Create trainer and start training
    trainer = MultiscaleGlobalWaveTrainer(config)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial results if available
        if hasattr(trainer, 'history') and trainer.history['train_loss']:
            print("\n💾 Saving partial results...")
            with open(trainer.output_dir / "partial_history.json", 'w') as f:
                json.dump(trainer.history, f)
    
    print("\n🎉 Multiscale global wave model training complete!")
    print(f"   Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()