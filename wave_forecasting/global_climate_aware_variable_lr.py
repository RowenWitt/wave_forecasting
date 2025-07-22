"""
FIXED Global Climate-Aware Variable Learning Rate Wave Model
Proper MWD circular handling to prevent NaN propagation
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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
import random

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import existing components
from config.base import GlobalDataConfig

@dataclass
class GlobalVariableLRConfig:
    """Global configuration extending proven Variable LR approach - FIXED"""
    
    # Temporal parameters (proven from Variable LR)
    sequence_length: int = 6           
    prediction_horizon: int = 4        
    
    # FIXED: Enhanced global architecture with correct feature count
    input_features: int = 18           # 15 base + 6 climate + 1 bathymetry (CORRECTED)
    hidden_dim: int = 384              # Increased from 256 for global complexity
    temporal_hidden_dim: int = 192     # Increased from 128 proportionally
    num_spatial_layers: int = 8        # Increased from 6 for global connectivity
    num_temporal_layers: int = 2       # Keep proven temporal depth
    
    # Attention (proven settings)
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    use_climate_attention: bool = True  # New: separate attention for climate features
    num_attention_heads: int = 8
    
    # Regularization (proven settings)
    dropout: float = 0.15
    spatial_dropout: float = 0.1
    temporal_dropout: float = 0.1
    
    # Training parameters (proven from Variable LR)
    num_epochs: int = 200
    batch_size: int = 6                # Reduced for global memory usage
    base_learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    
    # Variable-specific learning rates (proven successful)
    swh_lr_multiplier: float = 0.7      # Maintain excellent SWH performance
    mwd_lr_multiplier: float = 1.0      # Standard for circular MWD
    mwp_lr_multiplier: float = 1.3      # Higher for MWP improvement
    
    # Adaptive learning rate parameters
    lr_patience: int = 15               # Proven patience
    lr_factor: float = 0.8              # Proven factor
    min_lr_factor: float = 0.1          # Minimum LR threshold
    
    # Early stopping (critical lesson learned)
    early_stopping_patience: int = 50  # More aggressive early stopping
    
    # Variable-specific loss weights (proven)
    swh_loss_weight: float = 1.0
    mwd_loss_weight: float = 1.0        
    mwp_loss_weight: float = 1.0
    physics_loss_weight: float = 0.2
    
    # Data parameters
    validation_split: float = 0.2
    max_training_sequences: int = 2000  # Increased for global domain
    
    # Curriculum learning
    start_with_single_step: bool = True
    single_step_epochs: int = 200
    
    # FIXED: Climate and bathymetry features (corrected indices)
    climate_feature_indices: List[int] = None  # Indices of climate anomaly features
    bathymetry_feature_index: int = 17         # Index of bathymetry feature
    
    def __post_init__(self):
        if self.climate_feature_indices is None:
            # Climate features are indices 11-16 (6 features)
            self.climate_feature_indices = [11, 12, 13, 14, 15, 16]
        
        print(f"🔧 FIXED Configuration:")
        print(f"   Input features: {self.input_features} (15 base + 6 climate + 1 bathymetry)")
        print(f"   Climate indices: {self.climate_feature_indices}")
        print(f"   Bathymetry index: {self.bathymetry_feature_index}")

class CircularNormalizer:
    """FIXED: Robust normalizer for circular variables like wave direction (0-360°)"""
    
    def __init__(self):
        self.fitted = False
        self.mean_cos = 0.0
        self.mean_sin = 0.0
        self.std_cos = 1.0
        self.std_sin = 1.0
    
    def fit(self, angles_deg: np.ndarray):
        """Fit normalizer on circular data with NaN handling"""
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(angles_deg)
        if np.sum(valid_mask) == 0:
            print("⚠️ Warning: All angles are NaN, using default normalization")
            self.fitted = True
            return
        
        valid_angles = angles_deg[valid_mask]
        
        # Convert to radians
        angles_rad = np.deg2rad(valid_angles)
        
        # Convert to unit circle
        cos_vals = np.cos(angles_rad)
        sin_vals = np.sin(angles_rad)
        
        # Compute statistics
        self.mean_cos = np.mean(cos_vals)
        self.mean_sin = np.mean(sin_vals)
        self.std_cos = np.std(cos_vals)
        self.std_sin = np.std(sin_vals)
        
        # Avoid division by zero
        self.std_cos = max(self.std_cos, 1e-6)
        self.std_sin = max(self.std_sin, 1e-6)
        
        self.fitted = True
        
        print(f"CircularNormalizer fitted on {len(valid_angles)} valid samples:")
        print(f"  Mean direction: {np.rad2deg(np.arctan2(self.mean_sin, self.mean_cos)):.1f}°")
        print(f"  Std cos: {self.std_cos:.3f}, sin: {self.std_sin:.3f}")
    
    def transform(self, angles_deg: np.ndarray) -> np.ndarray:
        """Transform angles to normalized [cos, sin] with NaN preservation"""
        if not self.fitted:
            raise ValueError("CircularNormalizer not fitted")
        
        angles_deg = angles_deg.flatten()
        
        # Preserve NaN mask
        nan_mask = np.isnan(angles_deg)
        
        # Create output array
        result = np.zeros((len(angles_deg), 2))
        result[nan_mask] = np.nan
        
        if np.sum(~nan_mask) > 0:
            valid_angles = angles_deg[~nan_mask]
            
            # Convert to radians
            angles_rad = np.deg2rad(valid_angles)
            
            # Convert to unit circle
            cos_vals = np.cos(angles_rad)
            sin_vals = np.sin(angles_rad)
            
            # Normalize
            cos_norm = (cos_vals - self.mean_cos) / self.std_cos
            sin_norm = (sin_vals - self.mean_sin) / self.std_sin
            
            # Store results
            result[~nan_mask, 0] = cos_norm
            result[~nan_mask, 1] = sin_norm
        
        return result
    
    def inverse_transform(self, normalized_circular: np.ndarray) -> np.ndarray:
        """Transform normalized [cos, sin] back to angles with NaN handling"""
        if not self.fitted:
            raise ValueError("CircularNormalizer not fitted")
        
        # Handle NaN values
        nan_mask = np.isnan(normalized_circular).any(axis=1)
        result = np.full(len(normalized_circular), np.nan)
        
        if np.sum(~nan_mask) > 0:
            valid_data = normalized_circular[~nan_mask]
            
            # Denormalize
            cos_vals = valid_data[:, 0] * self.std_cos + self.mean_cos
            sin_vals = valid_data[:, 1] * self.std_sin + self.mean_sin
            
            # Convert back to angles
            angles_rad = np.arctan2(sin_vals, cos_vals)
            angles_deg = np.rad2deg(angles_rad)
            
            # Ensure [0, 360) range
            angles_deg = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
            
            result[~nan_mask] = angles_deg
        
        return result

class VariableSpecificNormalizer:
    """FIXED: Separate normalizers with robust circular MWD handling"""
    
    def __init__(self):
        self.swh_scaler = StandardScaler()
        self.mwd_normalizer = CircularNormalizer()
        self.mwp_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, targets: np.ndarray):
        """Fit normalizers on wave data with NaN handling"""
        print(f"🔧 Fitting VariableSpecificNormalizer on {targets.shape} targets...")
        
        # Handle input validation
        if targets.shape[1] != 3:
            raise ValueError(f"Expected targets shape (N, 3), got {targets.shape}")
        
        swh = targets[:, 0:1]  # Keep 2D for sklearn
        mwd = targets[:, 1]    # 1D for circular normalizer
        mwp = targets[:, 2:3]  # Keep 2D for sklearn
        
        # Check for valid data
        swh_valid = ~np.isnan(swh).all(axis=1)
        mwp_valid = ~np.isnan(mwp).all(axis=1)
        
        print(f"   Valid samples: SWH={np.sum(swh_valid)}, MWD={np.sum(~np.isnan(mwd))}, MWP={np.sum(mwp_valid)}")
        
        # Fit each normalizer with valid data only
        if np.sum(swh_valid) > 0:
            self.swh_scaler.fit(swh[swh_valid])
        
        self.mwd_normalizer.fit(mwd)  # Handles NaN internally
        
        if np.sum(mwp_valid) > 0:
            self.mwp_scaler.fit(mwp[mwp_valid])
        
        self.fitted = True
        
        print(f"   SWH range: {np.nanmin(swh):.2f} to {np.nanmax(swh):.2f} m")
        print(f"   MWD range: {np.nanmin(mwd):.1f} to {np.nanmax(mwd):.1f} °")
        print(f"   MWP range: {np.nanmin(mwp):.2f} to {np.nanmax(mwp):.2f} s")
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets: [SWH, MWD, MWP] → [SWH_norm, MWD_cos_norm, MWD_sin_norm, MWP_norm]"""
        if not self.fitted:
            raise ValueError("VariableSpecificNormalizer not fitted")
        
        if targets.shape[1] != 3:
            raise ValueError(f"Expected targets shape (N, 3), got {targets.shape}")
        
        swh = targets[:, 0:1]
        mwd = targets[:, 1]
        mwp = targets[:, 2:3]
        
        # Transform each variable
        swh_norm = self.swh_scaler.transform(swh)  # [N, 1]
        mwd_norm = self.mwd_normalizer.transform(mwd)  # [N, 2] cos, sin
        mwp_norm = self.mwp_scaler.transform(mwp)  # [N, 1]
        
        # Concatenate: [SWH, MWD_cos, MWD_sin, MWP]
        result = np.concatenate([swh_norm, mwd_norm, mwp_norm], axis=1)
        
        # Final NaN check
        nan_count = np.sum(np.isnan(result))
        if nan_count > 0:
            print(f"⚠️ Warning: {nan_count} NaN values in transformed targets")
        
        return result
    
    def inverse_transform_targets(self, normalized: np.ndarray) -> np.ndarray:
        """Transform normalized targets back: [SWH_norm, MWD_cos_norm, MWD_sin_norm, MWP_norm] → [SWH, MWD, MWP]"""
        if not self.fitted:
            raise ValueError("VariableSpecificNormalizer not fitted")
        
        if normalized.shape[1] != 4:
            raise ValueError(f"Expected normalized shape (N, 4), got {normalized.shape}")
        
        swh_norm = normalized[:, 0:1]    # [N, 1]
        mwd_norm = normalized[:, 1:3]    # [N, 2] cos, sin
        mwp_norm = normalized[:, 3:4]    # [N, 1]
        
        # Denormalize each variable
        swh = self.swh_scaler.inverse_transform(swh_norm)       # [N, 1]
        mwd = self.mwd_normalizer.inverse_transform(mwd_norm)   # [N,] (1D)
        mwp = self.mwp_scaler.inverse_transform(mwp_norm)       # [N, 1]
        
        # Ensure consistent shapes and concatenate
        swh_flat = swh.flatten()
        mwp_flat = mwp.flatten()
        
        # Concatenate back to [SWH, MWD, MWP]
        return np.column_stack([swh_flat, mwd, mwp_flat])

class RobustCircularLoss(nn.Module):
    """FIXED: Loss function with robust circular MWD handling"""
    
    def __init__(self, config: GlobalVariableLRConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with robust circular MWD handling
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
        
        # Create valid masks (exclude NaN and extreme values)
        swh_valid = torch.isfinite(pred_swh) & torch.isfinite(true_swh)
        mwd_cos_valid = torch.isfinite(pred_mwd_cos) & torch.isfinite(true_mwd_cos)
        mwd_sin_valid = torch.isfinite(pred_mwd_sin) & torch.isfinite(true_mwd_sin)
        mwp_valid = torch.isfinite(pred_mwp) & torch.isfinite(true_mwp)
        
        # SWH Loss (standard MSE on valid values)
        if torch.sum(swh_valid) > 0:
            swh_loss = F.mse_loss(pred_swh[swh_valid], true_swh[swh_valid])
        else:
            swh_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # MWD Loss (robust circular handling)
        mwd_valid = mwd_cos_valid & mwd_sin_valid
        
        if torch.sum(mwd_valid) > 0:
            # Gently constrain cos/sin to reasonable range
            pred_mwd_cos_clipped = torch.clamp(pred_mwd_cos[mwd_valid], -3.0, 3.0)
            pred_mwd_sin_clipped = torch.clamp(pred_mwd_sin[mwd_valid], -3.0, 3.0)
            true_mwd_cos_valid = true_mwd_cos[mwd_valid]
            true_mwd_sin_valid = true_mwd_sin[mwd_valid]
            
            # Component losses
            mwd_cos_loss = F.mse_loss(pred_mwd_cos_clipped, true_mwd_cos_valid)
            mwd_sin_loss = F.mse_loss(pred_mwd_sin_clipped, true_mwd_sin_valid)
            
            # FIXED: Safe angular distance computation
            # Add small epsilon to prevent atan2(0,0) which causes NaN gradients
            eps = 1e-8
            pred_cos_safe = pred_mwd_cos_clipped + eps
            pred_sin_safe = pred_mwd_sin_clipped + eps
            true_cos_safe = true_mwd_cos_valid + eps
            true_sin_safe = true_mwd_sin_valid + eps
            
            # Normalize to ensure they're on unit circle (approximately)
            pred_norm = torch.sqrt(pred_cos_safe**2 + pred_sin_safe**2)
            pred_cos_safe = pred_cos_safe / pred_norm
            pred_sin_safe = pred_sin_safe / pred_norm
            
            true_norm = torch.sqrt(true_cos_safe**2 + true_sin_safe**2)
            true_cos_safe = true_cos_safe / true_norm
            true_sin_safe = true_sin_safe / true_norm
            
            # Use dot product for angular distance instead of atan2
            # cos(θ₁ - θ₂) = cos(θ₁)cos(θ₂) + sin(θ₁)sin(θ₂)
            cos_diff = pred_cos_safe * true_cos_safe + pred_sin_safe * true_sin_safe
            cos_diff = torch.clamp(cos_diff, -1.0 + eps, 1.0 - eps)  # Prevent acos(±1) issues
            
            # Angular loss: 1 - cos(θ₁ - θ₂) (always positive, smooth)
            mwd_angular_loss = torch.mean(1.0 - cos_diff)
            
            # Combined MWD loss with emphasis on component losses
            mwd_loss = 0.7 * (mwd_cos_loss + mwd_sin_loss) + 0.3 * mwd_angular_loss
        else:
            mwd_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            mwd_cos_loss = torch.tensor(0.0, device=predictions.device)
            mwd_sin_loss = torch.tensor(0.0, device=predictions.device)
            mwd_angular_loss = torch.tensor(0.0, device=predictions.device)
        
        # MWP Loss (standard MSE on valid values)
        if torch.sum(mwp_valid) > 0:
            mwp_loss = F.mse_loss(pred_mwp[mwp_valid], true_mwp[mwp_valid])
        else:
            mwp_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Physics constraints (gentle)
        physics_loss = self._compute_gentle_physics_constraints(predictions)
        
        # Weighted total loss
        total_loss = (self.config.swh_loss_weight * swh_loss +
                     self.config.mwd_loss_weight * mwd_loss +
                     self.config.mwp_loss_weight * mwp_loss +
                     self.config.physics_loss_weight * physics_loss)
        
        # Final safety check
        if not torch.isfinite(total_loss):
            print("⚠️ Warning: Non-finite total loss, using fallback")
            total_loss = swh_loss + mwp_loss
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_loss': mwd_loss,
            'mwp_loss': mwp_loss,
            'physics_loss': physics_loss,
            'mwd_circular_loss': mwd_cos_loss + mwd_sin_loss if 'mwd_cos_loss' in locals() else torch.tensor(0.0),
            'mwd_angular_loss': mwd_angular_loss if 'mwd_angular_loss' in locals() else torch.tensor(0.0)
        }
    
    def _compute_gentle_physics_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """Gentle physics constraints to guide learning"""
        pred_swh = predictions[:, :, 0]
        pred_mwp = predictions[:, :, 3]
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        
        # Create valid masks
        valid_mask = torch.isfinite(pred_swh) & torch.isfinite(pred_mwp)
        
        if torch.sum(valid_mask) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Gentle physical bounds
        swh_penalty = F.relu(-pred_swh[valid_mask]).mean() * 0.1  # Gentle SWH >= 0
        mwp_penalty = F.relu(1 - pred_mwp[valid_mask]).mean() * 0.1  # Gentle MWP >= 1
        
        # Very gentle unit circle constraint for MWD
        mwd_valid = torch.isfinite(pred_mwd_cos) & torch.isfinite(pred_mwd_sin)
        if torch.sum(mwd_valid) > 0:
            circle_deviation = torch.abs(pred_mwd_cos[mwd_valid]**2 + pred_mwd_sin[mwd_valid]**2 - 1)
            circle_penalty = torch.mean(circle_deviation) * 0.05
        else:
            circle_penalty = torch.tensor(0.0, device=predictions.device)
        
        return swh_penalty + mwp_penalty + circle_penalty

class StreamingGlobalDataset(Dataset):
    """FIXED: Memory-efficient streaming dataset with robust target processing"""
    
    def __init__(self, data_files: List[str], config: GlobalVariableLRConfig, 
                 validation_split: float = 0.2, is_validation: bool = False):
        
        self.data_files = data_files
        self.config = config
        self.is_validation = is_validation
        
        # Build sequence index without loading data
        self.sequence_index = self._build_sequence_index()
        
        # Split train/val
        total_sequences = len(self.sequence_index)
        val_size = int(total_sequences * validation_split)
        
        if is_validation:
            self.sequence_index = self.sequence_index[-val_size:]
        else:
            self.sequence_index = self.sequence_index[:-val_size]
        
        # Cache for loaded data
        self.data_cache = {}
        
        print(f"🌍 FIXED Global Streaming Dataset ({'validation' if is_validation else 'training'}):")
        print(f"   Sequences: {len(self.sequence_index)}")
        print(f"   Files: {len(self.data_files)}")
        print(f"   Features: {self.config.input_features}")
    
    def _build_sequence_index(self):
        """Build index of all sequences without loading data"""
        sequences = []
        
        for file_path in self.data_files:
            try:
                with xr.open_dataset(file_path) as ds:
                    print(f"   📂 Indexing {Path(file_path).name}")
                    
                    # Find time coordinate
                    time_coord = None
                    for coord in ['time', 'valid_time', 'forecast_time', 't']:
                        if coord in ds.coords:
                            time_coord = coord
                            break
                    
                    if time_coord is None:
                        time_coords_found = [coord for coord in ds.coords if 'time' in coord.lower()]
                        if time_coords_found:
                            time_coord = time_coords_found[0]
                    
                    if time_coord is None:
                        print(f"      ❌ No time coordinate found")
                        continue
                    
                    time_steps = len(ds[time_coord])
                    print(f"      Time steps: {time_steps}")
                    
                    # Build sequences for this file
                    file_sequences = 0
                    for t in range(time_steps - self.config.sequence_length - 1):
                        sequences.append({
                            'file': file_path,
                            'time_idx': t,
                            'file_timesteps': time_steps,
                            'time_coord': time_coord
                        })
                        file_sequences += 1
                    
                    print(f"      ✅ Created {file_sequences} sequences")
                    
            except Exception as e:
                print(f"      ❌ Could not index {file_path}: {e}")
                continue
        
        print(f"   📊 Total sequences found: {len(sequences)}")
        return sequences
    
    def _load_file_data(self, file_path: str):
        """Load and cache file data with robust feature extraction"""
        if file_path in self.data_cache:
            return self.data_cache[file_path]
        
        try:
            ds = xr.open_dataset(file_path)
            
            print(f"   📂 Loading {Path(file_path).name}")
            
            # Find time coordinate
            time_coord = None
            for coord in ['time', 'valid_time', 'forecast_time', 't']:
                if coord in ds.coords:
                    time_coord = coord
                    break
            
            if time_coord is None:
                time_coords_found = [coord for coord in ds.coords if 'time' in coord.lower()]
                if time_coords_found:
                    time_coord = time_coords_found[0]
            
            if time_coord is None:
                print(f"      ❌ No time coordinate found")
                return None
            
            # Check for required features
            has_anomalies = any('anomaly' in var for var in ds.data_vars)
            has_bathymetry = 'ocean_depth' in ds.data_vars
            
            if not has_anomalies or not has_bathymetry:
                print(f"      ⚠️ Missing required features (anomalies: {has_anomalies}, bathymetry: {has_bathymetry})")
                return None
            
            # Priority feature selection for exactly 18 features
            all_vars = list(ds.data_vars.keys())
            
            # Define priority order
            priority_vars = [
                # Base atmospheric (5)
                'u10', 'v10', 'msl', 'sst', 'tp',
                
                # Wave variables (4)
                'swh', 'mwd', 'mwp', 'shww',
                
                # Upper air (6)
                'u_850', 'v_850', 'z_500', 'u_500', 'v_500', 'z_850',
                
                # Climate anomalies (6)
                'sst_anomaly', 'slp_anomaly', 'u850_anomaly', 
                'v850_anomaly', 'precip_anomaly', 'z500_anomaly',
                
                # Bathymetry (1)
                'ocean_depth'
            ]
            
            # Select available features in priority order
            selected_vars = []
            for var in priority_vars:
                if var in all_vars:
                    selected_vars.append(var)
                if len(selected_vars) >= 18:
                    break
            
            # Fill remaining slots if needed
            if len(selected_vars) < 18:
                for var in all_vars:
                    if var not in selected_vars and len(selected_vars) < 18:
                        selected_vars.append(var)
            
            # Ensure exactly 18 features
            if len(selected_vars) < 18:
                # Duplicate features to reach 18
                while len(selected_vars) < 18:
                    selected_vars.append(selected_vars[-1])
            else:
                selected_vars = selected_vars[:18]
            
            print(f"      Selected 18 features: {selected_vars[:5]}... + {len(selected_vars)-5} more")
            
            # Extract feature arrays
            feature_arrays = []
            feature_names = []
            
            for var in selected_vars:
                try:
                    feature_arrays.append(ds[var].values)
                    feature_names.append(var)
                except Exception as e:
                    print(f"         ⚠️ Error loading {var}: {e}")
                    # Use zeros as fallback
                    ref_shape = ds[list(ds.data_vars.keys())[0]].shape
                    feature_arrays.append(np.zeros(ref_shape))
                    feature_names.append(f"{var}_fallback")
            
            # Stack features
            combined_features = np.stack(feature_arrays, axis=-1)  # [time, lat, lon, 18]
            
            print(f"      ✅ Feature array: {combined_features.shape}")
            
            # Handle coordinates
            if 'lat' in ds.coords and 'lon' in ds.coords:
                coords = {'lat': ds.lat.values, 'lon': ds.lon.values}
            elif 'latitude' in ds.coords and 'longitude' in ds.coords:
                coords = {'lat': ds.latitude.values, 'lon': ds.longitude.values}
            else:
                print(f"      ❌ No recognizable spatial coordinates found")
                return None
            
            cached_data = {
                'features': combined_features,
                'feature_names': feature_names,
                'coords': {
                    'time': ds[time_coord].values,
                    'lat': coords['lat'],
                    'lon': coords['lon']
                },
                'time_coord': time_coord
            }
            
            self.data_cache[file_path] = cached_data
            ds.close()
            
            print(f"      ✅ Cached successfully")
            return cached_data
            
        except Exception as e:
            print(f"      ❌ Error loading {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.sequence_index)
    
    def __getitem__(self, idx):
        """FIXED: Get sequence with robust target processing and NaN handling"""
        seq_info = self.sequence_index[idx]
        
        # Load file data
        file_data = self._load_file_data(seq_info['file'])
        if file_data is None:
            return self._get_dummy_sequence()
        
        # Extract sequence with bounds checking
        t_start = seq_info['time_idx']
        t_end = t_start + self.config.sequence_length
        
        try:
            # Bounds checking
            max_time = file_data['features'].shape[0]
            if t_end >= max_time:
                return self._get_dummy_sequence()
            
            # Input sequence
            input_sequence = file_data['features'][t_start:t_end]  # [seq_len, lat, lon, 18]
            
            # Target (next timestep)
            if t_end < max_time:
                target_timestep = file_data['features'][t_end]  # [lat, lon, 18]
            else:
                return self._get_dummy_sequence()
            
            # Validate shapes
            if input_sequence.ndim != 4 or target_timestep.ndim != 3:
                return self._get_dummy_sequence()
            
            if input_sequence.shape[-1] != 18:
                print(f"Warning: Expected 18 features, got {input_sequence.shape[-1]}")
                return self._get_dummy_sequence()
            
            # FIXED: Extract wave variables as targets with robust name matching
            feature_names = file_data.get('feature_names', [])
            
            # Find indices for SWH, MWD, MWP with flexible matching
            swh_idx = self._find_variable_index(feature_names, ['swh', 'significant_height', 'Hs'])
            mwd_idx = self._find_variable_index(feature_names, ['mwd', 'mean_wave_direction', 'wave_direction'])
            mwp_idx = self._find_variable_index(feature_names, ['mwp', 'mean_wave_period', 'wave_period'])
            
            if swh_idx is None or mwd_idx is None or mwp_idx is None:
                print(f"Warning: Cannot find wave variables")
                print(f"   Available features: {feature_names}")
                print(f"   SWH: {swh_idx}, MWD: {mwd_idx}, MWP: {mwp_idx}")
                return self._get_dummy_sequence()
            
            # Extract targets
            targets = target_timestep[:, :, [swh_idx, mwd_idx, mwp_idx]]  # [lat, lon, 3]
            
            # FIXED: Create robust ocean mask
            swh_target = targets[:, :, 0]
            
            # Multiple criteria for ocean points
            ocean_mask = (
                ~np.isnan(swh_target) &  # Not NaN
                (swh_target > 0.01) &    # Reasonable SWH values
                (swh_target < 30.0)      # Not extreme values
            )
            
            # Check MWD validity too
            mwd_target = targets[:, :, 1]
            mwd_valid = (
                ~np.isnan(mwd_target) &
                (mwd_target >= 0) &
                (mwd_target <= 360)
            )
            
            # Combine masks
            ocean_mask = ocean_mask & mwd_valid
            
            ocean_points = np.sum(ocean_mask)
            if ocean_points < 50:
                print(f"Warning: Too few valid ocean points ({ocean_points})")
                return self._get_dummy_sequence()
            
            # Flatten spatial dimensions
            seq_len, height, width, n_features = input_sequence.shape
            
            input_flat = input_sequence.reshape(seq_len, -1, n_features)
            target_flat = targets.reshape(-1, 3)
            ocean_mask_flat = ocean_mask.flatten()
            
            # Select only ocean points
            ocean_indices = np.where(ocean_mask_flat)[0]
            input_ocean = input_flat[:, ocean_indices, :]
            target_ocean = target_flat[ocean_indices, :]
            
            # Subsample if too many points (memory management)
            max_nodes = 1500
            if len(target_ocean) > max_nodes:
                sample_indices = np.random.choice(len(target_ocean), max_nodes, replace=False)
                input_ocean = input_ocean[:, sample_indices, :]
                target_ocean = target_ocean[sample_indices, :]
            elif len(target_ocean) < 50:
                # If too few, repeat some points
                repeat_factor = max(1, 50 // len(target_ocean))
                if repeat_factor > 1:
                    input_ocean = np.tile(input_ocean, (1, repeat_factor, 1))
                    target_ocean = np.tile(target_ocean, (repeat_factor, 1))
            
            # FIXED: Final data validation and cleaning
            input_ocean = self._clean_array(input_ocean, "input")
            target_ocean = self._clean_target_array(target_ocean, "target")
            
            # Convert to tensors
            input_tensor = torch.tensor(input_ocean, dtype=torch.float32)
            target_tensor = torch.tensor(target_ocean, dtype=torch.float32)
            
            # Final validation
            if torch.isnan(input_tensor).any():
                print("Warning: NaN in input tensor after cleaning")
                return self._get_dummy_sequence()
            
            if torch.isnan(target_tensor).any():
                print("Warning: NaN in target tensor after cleaning")
                return self._get_dummy_sequence()
            
            return {
                'input': input_tensor,
                'single_step_target': target_tensor
            }
            
        except Exception as e:
            print(f"Error extracting sequence {idx}: {e}")
            return self._get_dummy_sequence()
    
    def _find_variable_index(self, feature_names: List[str], possible_names: List[str]) -> Optional[int]:
        """Find variable index with flexible name matching"""
        for i, name in enumerate(feature_names):
            for possible in possible_names:
                if possible.lower() in name.lower():
                    return i
        return None
    
    def _clean_array(self, arr: np.ndarray, name: str) -> np.ndarray:
        """Clean array of NaN and extreme values"""
        if np.isnan(arr).any():
            print(f"   Cleaning NaN values in {name}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip extreme values
        arr = np.clip(arr, -1e6, 1e6)
        
        return arr
    
    def _clean_target_array(self, targets: np.ndarray, name: str) -> np.ndarray:
        """Clean target array with variable-specific constraints"""
        targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Variable-specific cleaning
        # SWH: 0-30m reasonable
        targets[:, 0] = np.clip(targets[:, 0], 0.01, 30.0)
        
        # MWD: 0-360°
        targets[:, 1] = np.clip(targets[:, 1], 0.0, 360.0)
        
        # MWP: 1-25s reasonable
        targets[:, 2] = np.clip(targets[:, 2], 1.0, 25.0)
        
        return targets
    
    def _get_dummy_sequence(self):
        """Return dummy sequence for error cases"""
        dummy_nodes = 1000
        
        input_tensor = torch.zeros(self.config.sequence_length, dummy_nodes, 18)
        target_tensor = torch.zeros(dummy_nodes, 3)
        
        # Set reasonable dummy values
        target_tensor[:, 0] = 1.0    # SWH = 1m
        target_tensor[:, 1] = 180.0  # MWD = 180°
        target_tensor[:, 2] = 8.0    # MWP = 8s
        
        return {
            'input': input_tensor,
            'single_step_target': target_tensor
        }

class GlobalVariableLRSpatioTemporalGNN(nn.Module):
    """FIXED: Global climate-aware model with robust MWD handling"""
    
    def __init__(self, config: GlobalVariableLRConfig):
        super().__init__()
        self.config = config
        
        print(f"🌍 FIXED Global Climate-Aware Variable LR Model:")
        print(f"   Input features: {config.input_features}")
        print(f"   Hidden dim: {config.hidden_dim}")
        print(f"   Robust MWD handling: ✅")
        
        # Enhanced feature encoding
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Climate feature processing
        self.climate_processor = nn.Sequential(
            nn.Linear(6, 64),  # 6 climate features
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Bathymetry processing
        self.bathymetry_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Spatial layers
        self.spatial_layers = nn.ModuleList([
            EnhancedMessageLayer(config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Attention mechanisms
        if config.use_spatial_attention:
            self.spatial_attention = SpatialAttention(config.hidden_dim, config.num_attention_heads)
        
        if config.use_climate_attention:
            self.climate_attention = ClimateAttention(config.hidden_dim, climate_dim=6)
        
        # Temporal processing
        self.temporal_encoder = nn.LSTM(
            config.hidden_dim,
            config.temporal_hidden_dim,
            config.num_temporal_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.num_temporal_layers > 1 else 0
        )
        
        if config.use_temporal_attention:
            self.temporal_attention = TemporalAttention(config.temporal_hidden_dim, config.num_attention_heads)
        
        # Shared feature transformation
        hidden_size = config.hidden_dim // 2
        self.shared_head = nn.Sequential(
            nn.Linear(config.temporal_hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # FIXED: Variable-specific heads with proper output sizes
        self.swh_head = nn.Linear(hidden_size, 1)      # SWH output
        self.mwd_head = nn.Linear(hidden_size, 2)      # MWD [cos, sin] output  
        self.mwp_head = nn.Linear(hidden_size, 1)      # MWP output
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling to prevent NaN gradients"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Much smaller initialization to prevent gradient explosion
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Reduced from 0.8
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.1)  # Reduced
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.1)      # Reduced
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                multi_step: bool = False) -> torch.Tensor:
        """FIXED: Forward pass with robust processing"""
        
        batch_size, seq_len, num_nodes, input_features = x.size()
        
        if input_features != self.config.input_features:
            raise ValueError(f"Expected {self.config.input_features} features, got {input_features}")
        
        # Process each timestep
        spatial_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, 18]
            
            # Extract climate and bathymetry features safely
            if len(self.config.climate_feature_indices) > 0:
                climate_features = x_t[:, :, self.config.climate_feature_indices]
            else:
                climate_features = torch.zeros(batch_size, num_nodes, 6, device=x.device)
            
            if self.config.bathymetry_feature_index < input_features:
                bathymetry_feature = x_t[:, :, self.config.bathymetry_feature_index:self.config.bathymetry_feature_index+1]
            else:
                bathymetry_feature = torch.zeros(batch_size, num_nodes, 1, device=x.device)
            
            # Main feature encoding
            h_t = self.feature_encoder(x_t)
            
            # Process additional features
            climate_processed = self.climate_processor(climate_features)
            bathy_processed = self.bathymetry_processor(bathymetry_feature)
            
            # Spatial processing for each sample in batch
            batch_outputs = []
            for b in range(batch_size):
                h_b = h_t[b]
                climate_b = climate_features[b]
                
                # Message passing
                for layer in self.spatial_layers:
                    h_b = layer(h_b, edge_index, edge_attr)
                
                # Spatial attention
                if hasattr(self, 'spatial_attention'):
                    h_b = self.spatial_attention(h_b.unsqueeze(0), edge_index).squeeze(0)
                
                # Climate attention
                if hasattr(self, 'climate_attention'):
                    h_b = self.climate_attention(h_b.unsqueeze(0), climate_b.unsqueeze(0)).squeeze(0)
                
                batch_outputs.append(h_b)
            
            h_t = torch.stack(batch_outputs, dim=0)
            spatial_outputs.append(h_t)
        
        # Stack temporal sequence
        spatial_sequence = torch.stack(spatial_outputs, dim=1)
        
        # FIXED: Stable temporal processing to replace problematic LSTM
        # Instead of per-node LSTM which is causing instability, use global approach
        
        # Reshape for batch processing: [batch*nodes, seq_len, hidden_dim]
        batch_size, seq_len, num_nodes, hidden_dim = spatial_sequence.shape
        sequence_reshaped = spatial_sequence.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, hidden_dim)
        
        # Clamp inputs to LSTM
        sequence_reshaped = torch.clamp(sequence_reshaped, -5, 5)
        
        # Use LSTM in batch mode (more stable)
        try:
            lstm_out, _ = self.temporal_encoder(sequence_reshaped)
            
            # Immediately clamp LSTM outputs  
            lstm_out = torch.clamp(lstm_out, -10, 10)
            
            # Check for NaN/Inf and fix
            if torch.isnan(lstm_out).any() or torch.isinf(lstm_out).any():
                print(f"   ⚠️ NaN/Inf detected in LSTM output - replacing with zeros")
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=5.0, neginf=-5.0)
            
            # Temporal attention if available
            if hasattr(self, 'temporal_attention'):
                lstm_out = self.temporal_attention(lstm_out)
                lstm_out = torch.clamp(lstm_out, -10, 10)
            
            # Get final states (last timestep)
            final_states = lstm_out[:, -1, :]  # [batch*nodes, hidden_dim]
            
            # Reshape back to [batch, nodes, hidden_dim]
            temporal_features = final_states.reshape(batch_size, num_nodes, -1)
            
        except Exception as e:
            print(f"   ❌ LSTM processing failed: {e}")
            # Fallback: use mean of spatial features
            temporal_features = torch.mean(spatial_sequence, dim=1)  # [batch, nodes, hidden_dim]
        
        # Shared feature transformation
        shared_features = self.shared_head(temporal_features)
        
        # FIXED: Variable-specific predictions with enhanced constraints
        swh_pred = self.swh_head(shared_features)  # [batch, nodes, 1]
        swh_pred = F.softplus(swh_pred)  # Ensure positive SWH
        swh_pred = torch.clamp(swh_pred, 0.01, 50)  # Reasonable bounds
        
        mwd_pred = self.mwd_head(shared_features)  # [batch, nodes, 2]
        # Clamp before normalization to prevent extreme values
        mwd_pred = torch.clamp(mwd_pred, -10, 10)
        # Normalize to unit circle (gently)
        mwd_norm = torch.norm(mwd_pred, dim=-1, keepdim=True)
        mwd_pred = mwd_pred / torch.clamp(mwd_norm, min=1e-6)
        
        mwp_pred = self.mwp_head(shared_features)  # [batch, nodes, 1]
        mwp_pred = F.softplus(mwp_pred) + 1.0  # Ensure MWP >= 1
        mwp_pred = torch.clamp(mwp_pred, 1.0, 30.0)  # Reasonable bounds
        
        # Combine predictions: [SWH, MWD_cos, MWD_sin, MWP]
        predictions = torch.cat([swh_pred, mwd_pred, mwp_pred], dim=-1)
        
        return predictions
    
    def get_variable_parameters(self):
        """Get parameters grouped by variable - SIMPLE WORKING VERSION"""
        
        print("🔧 Getting variable parameters (simple version)...")
        
        # Get ALL model parameters first
        all_params = list(self.parameters())
        print(f"   Total model parameters: {len(all_params)}")
        
        if len(all_params) == 0:
            print("   ❌ Model has no parameters!")
            return None
        
        # Try to get variable-specific parameters
        try:
            swh_params = list(self.swh_head.parameters())
            print(f"   SWH head parameters: {len(swh_params)}")
        except Exception as e:
            print(f"   ❌ SWH head error: {e}")
            return None
        
        try:
            mwd_params = list(self.mwd_head.parameters())
            print(f"   MWD head parameters: {len(mwd_params)}")
        except Exception as e:
            print(f"   ❌ MWD head error: {e}")
            return None
        
        try:
            mwp_params = list(self.mwp_head.parameters())
            print(f"   MWP head parameters: {len(mwp_params)}")
        except Exception as e:
            print(f"   ❌ MWP head error: {e}")
            return None
        
        # Create shared parameters (everything except the heads)
        head_params = set(swh_params + mwd_params + mwp_params)
        shared_params = [p for p in all_params if p not in head_params]
        
        print(f"   Shared parameters: {len(shared_params)}")
        
        # Validate we have everything
        if len(shared_params) == 0:
            print("   ❌ No shared parameters!")
            return None
        
        if len(swh_params) == 0 or len(mwd_params) == 0 or len(mwp_params) == 0:
            print("   ❌ Missing variable-specific parameters!")
            return None
        
        result = {
            'shared': shared_params,
            'swh': swh_params,
            'mwd': mwd_params,
            'mwp': mwp_params
        }
        
        print("   ✅ Parameter grouping successful!")
        return result
        
class StableLayerNorm(nn.LayerNorm):
    """Numerically stable LayerNorm to prevent NaN gradients"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-5  # Increase epsilon for stability
    
    def forward(self, x):
        # Clamp inputs to prevent extreme values
        x = torch.clamp(x, -50, 50)
        return super().forward(x)

def stable_softmax(scores, dim=-1):
    """Numerically stable softmax to prevent NaN gradients"""
    # Subtract max for numerical stability
    scores_max = torch.max(scores, dim=dim, keepdim=True)[0]
    scores_stable = scores - scores_max
    
    # Clamp to prevent overflow
    scores_stable = torch.clamp(scores_stable, -50, 50)
    
    return F.softmax(scores_stable, dim=dim)

# Support classes for the model
class ClimateAttention(nn.Module):
    """Climate-aware attention mechanism"""
    
    def __init__(self, hidden_dim: int, climate_dim: int = 6, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.climate_dim = climate_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.climate_projection = nn.Linear(climate_dim, hidden_dim)
        self.climate_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = StableLayerNorm(hidden_dim)
    
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
    """Spatial attention mechanism"""
    
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
        self.layer_norm = StableLayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if edge_index is not None and edge_index.numel() > 0:
            mask = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=torch.bool)
            if edge_index.shape[1] > 0:  # Check if there are edges
                mask[edge_index[0], edge_index[1]] = True
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = mask.expand(batch_size, self.num_heads, -1, -1)
                scores = scores.masked_fill(~mask, float('-inf'))
        
        # Use stable softmax instead of regular softmax
        attention_weights = stable_softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        
        output = self.out_linear(attended)
        return self.layer_norm(x + output)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    
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
        self.layer_norm = StableLayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Use stable softmax
        attention_weights = stable_softmax(scores, dim=-1)
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
        
        self.layer_norm = StableLayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            # No edges, return input unchanged
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

class GlobalVariableLearningManager:
    """Learning rate manager for variable-specific optimization"""
    
    def __init__(self, config: GlobalVariableLRConfig):
        self.config = config
        self.base_lr = config.base_learning_rate
        
        # Current learning rates
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
        print(f"   SWH LR: {self.current_lrs['swh']:.2e}")
        print(f"   MWD LR: {self.current_lrs['mwd']:.2e}")
        print(f"   MWP LR: {self.current_lrs['mwp']:.2e}")
    
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
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

