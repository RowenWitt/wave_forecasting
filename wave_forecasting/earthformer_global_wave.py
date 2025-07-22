#!/usr/bin/env python3
"""
EarthFormer Global Wave Model - Benchmark Implementation
Matches experimental setup of GNN model for fair comparison
Based on EarthFormer architecture (Gao et al., 2022)
"""

import os
import sys
import time
import math
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Set memory limit for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION (Matching GNN setup)
# ==============================================================================

@dataclass
class EarthFormerConfig:
    """Configuration for EarthFormer wave prediction model"""
    
    # Data paths (same as GNN)
    data_path: str = "data/v1_global/processed/v1_era5_with_anomalies_202101.nc"
    output_dir: str = "experiments/earthformer_wave_v1"
    
    # Data parameters (matching GNN)
    lat_size: int = 361  # 0.5 degree resolution
    lon_size: int = 720  # 0.5 degree resolution
    
    # Input features (same as GNN)
    input_features: List[str] = field(default_factory=lambda: [
        'tp', 'u10', 'v10', 'msl', 'sst',  # Surface atmospheric (5)
        'swh', 'mwd', 'mwp', 'shww',       # Wave variables (4)
        'u_850', 'u_500', 'v_850', 'v_500', 'z_850', 'z_500',  # Pressure levels (6)
        'ocean_depth'  # Bathymetry (1)
    ])
    num_input_features: int = 16
    
    # Target variables (same as GNN)
    target_features: List[str] = field(default_factory=lambda: ['swh', 'mwd', 'mwp'])
    num_output_features: int = 4  # SWH + MWD(cos,sin) + MWP
    
    # Temporal parameters (same as GNN)
    sequence_length: int = 4  # 24 hours of input
    prediction_horizon: int = 1  # Single step prediction for fair comparison
    
    # EarthFormer architecture parameters
    patch_size: int = 4  # Patch size for vision transformer
    dim: int = 128  # Hidden dimension (matching GNN hidden_dim)
    depth: int = 4  # Number of transformer blocks (matching GNN layers)
    heads: int = 4  # Attention heads (matching GNN)
    mlp_ratio: int = 4  # MLP expansion ratio
    
    # Cuboid attention parameters
    window_size: int = 8  # Spatial window for local attention
    
    # Training parameters (same as GNN)
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    dropout: float = 0.15
    
    # Early stopping (same as GNN)
    early_stopping_patience: int = 20
    validation_split: float = 0.2
    
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# ==============================================================================
# EARTHFORMER COMPONENTS
# ==============================================================================

class CuboidSelfAttention(nn.Module):
    """Cuboid self-attention for spatiotemporal data"""
    
    def __init__(self, dim, heads, window_size, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, H, W, C = x.shape
        
        # Window partition
        x_windows = self.window_partition(x)  # (B*num_windows, window_size, window_size, T, C)
        B_win = x_windows.shape[0]
        
        # QKV
        qkv = self.qkv(x_windows).reshape(B_win, -1, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B_win, -1, C)
        out = self.proj(out)
        
        # Reverse window partition
        out = self.window_reverse(out, T, H, W)
        
        return out
    
    def window_partition(self, x):
        """Partition into non-overlapping windows"""
        B, T, H, W, C = x.shape
        x = x.view(B, T, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        windows = windows.view(-1, T * self.window_size * self.window_size, C)
        return windows
    
    def window_reverse(self, windows, T, H, W):
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, T, self.window_size, self.window_size, -1)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
        return x

class EarthFormerBlock(nn.Module):
    """EarthFormer transformer block"""
    
    def __init__(self, dim, heads, window_size, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CuboidSelfAttention(dim, heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EarthFormer(nn.Module):
    """EarthFormer model for global wave prediction"""
    
    def __init__(self, config: EarthFormerConfig):
        super().__init__()
        self.config = config
        
        # Calculate patch grid size
        self.h_patches = config.lat_size // config.patch_size
        self.w_patches = config.lon_size // config.patch_size
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv3d(config.num_input_features, config.dim, 
                     kernel_size=(1, config.patch_size, config.patch_size),
                     stride=(1, config.patch_size, config.patch_size)),
            Rearrange('b c t h w -> b t h w c')
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.sequence_length, self.h_patches, self.w_patches, config.dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EarthFormerBlock(
                config.dim, config.heads, config.window_size, 
                config.mlp_ratio, config.dropout
            ) for _ in range(config.depth)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim // 2, config.num_output_features * config.patch_size * config.patch_size),
            Rearrange('b t h w (c p1 p2) -> b c t (h p1) (w p2)', 
                      c=config.num_output_features, 
                      p1=config.patch_size, 
                      p2=config.patch_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: (B, T, H, W, C) -> rearrange to (B, C, T, H, W) for conv
        x = rearrange(x, 'b t h w c -> b c t h w')
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.norm(x)
        x = self.head(x)
        
        # Extract single timestep prediction
        x = x[:, :, -1, :, :]  # (B, C, H, W)
        
        return x

# ==============================================================================
# DATA LOADING (Reuse from GNN with modifications)
# ==============================================================================

class EarthFormerWaveDataset(Dataset):
    """Dataset for EarthFormer wave prediction"""
    
    def __init__(self, data_path: str, config: EarthFormerConfig, 
                 start_idx: int = 0, end_idx: int = None):
        self.data_path = data_path
        self.config = config
        
        print(f"📊 Loading wave data from: {data_path}")
        self.data = xr.open_dataset(data_path)
        
        # Time handling
        time_dim = 'time' if 'time' in self.data.dims else 'valid_time'
        self.time_steps = len(self.data[time_dim])
        
        # Sequence indices
        self.start_idx = start_idx
        self.end_idx = end_idx or (self.time_steps - config.sequence_length - config.prediction_horizon)
        
        print(f"✅ Dataset initialized: {self.end_idx - self.start_idx} sequences")
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # Extract input features for sequence
        input_features = []
        
        for t in range(self.config.sequence_length):
            t_idx = actual_idx + t
            timestep_features = []
            
            for feat in self.config.input_features:
                if feat in self.data.variables:
                    if 'time' in self.data[feat].dims:
                        field_data = self.data[feat].isel(time=t_idx).values
                    elif 'valid_time' in self.data[feat].dims:
                        field_data = self.data[feat].isel(valid_time=t_idx).values
                    else:
                        field_data = self.data[feat].values
                    
                    timestep_features.append(field_data)
                else:
                    # Feature not found, use zeros
                    timestep_features.append(np.zeros((self.config.lat_size, self.config.lon_size)))
            
            input_features.append(np.stack(timestep_features, axis=-1))
        
        # Stack temporal sequence: (T, H, W, C)
        inputs = np.stack(input_features, axis=0)
        
        # Extract targets (single timestep)
        target_idx = actual_idx + self.config.sequence_length
        target_features = []
        
        for feat in self.config.target_features:
            if feat in self.data.variables:
                if 'time' in self.data[feat].dims:
                    field_data = self.data[feat].isel(time=target_idx).values
                elif 'valid_time' in self.data[feat].dims:
                    field_data = self.data[feat].isel(valid_time=target_idx).values
                else:
                    field_data = self.data[feat].values
                
                target_features.append(field_data)
        
        targets = np.stack(target_features, axis=-1)  # (H, W, 3)
        
        # Handle NaN values
        inputs = np.nan_to_num(inputs, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
        
        return {
            'input': torch.FloatTensor(inputs),
            'target': torch.FloatTensor(targets)
        }

# ==============================================================================
# TRAINING (Adapted from GNN)
# ==============================================================================

# Reuse normalizers from GNN
class CircularNormalizer:
    def __init__(self):
        self.fitted = False
    
    def fit(self, angles_deg: np.ndarray):
        self.fitted = True
    
    def transform(self, angles_deg: np.ndarray) -> np.ndarray:
        angles_rad = np.deg2rad(angles_deg)
        cos_vals = np.cos(angles_rad)
        sin_vals = np.sin(angles_rad)
        return np.column_stack([cos_vals, sin_vals])
    
    def inverse_transform(self, cos_sin: np.ndarray) -> np.ndarray:
        angles_rad = np.arctan2(cos_sin[:, 1], cos_sin[:, 0])
        angles_deg = np.rad2deg(angles_rad)
        return np.where(angles_deg < 0, angles_deg + 360, angles_deg)

class VariableSpecificNormalizer:
    def __init__(self):
        self.swh_scaler = RobustScaler()
        self.mwd_normalizer = CircularNormalizer()
        self.mwp_scaler = RobustScaler()
        self.fitted = False
    
    def fit(self, targets: np.ndarray):
        swh = targets[..., 0].reshape(-1, 1)
        mwd = targets[..., 1].reshape(-1)
        mwp = targets[..., 2].reshape(-1, 1)
        
        self.swh_scaler.fit(swh)
        self.mwd_normalizer.fit(mwd)
        self.mwp_scaler.fit(mwp)
        self.fitted = True
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        shape = targets.shape
        targets_flat = targets.reshape(-1, 3)
        
        swh = targets_flat[:, 0:1]
        mwd = targets_flat[:, 1]
        mwp = targets_flat[:, 2:3]
        
        swh_norm = self.swh_scaler.transform(swh)
        mwd_norm = self.mwd_normalizer.transform(mwd)
        mwp_norm = self.mwp_scaler.transform(mwp)
        
        result = np.concatenate([swh_norm, mwd_norm, mwp_norm], axis=1)
        return result.reshape(shape[:-1] + (4,))
    
    def inverse_transform_targets(self, normalized: np.ndarray) -> np.ndarray:
        shape = normalized.shape
        norm_flat = normalized.reshape(-1, 4)
        
        swh_norm = norm_flat[:, 0:1]
        mwd_norm = norm_flat[:, 1:3]
        mwp_norm = norm_flat[:, 3:4]
        
        swh = self.swh_scaler.inverse_transform(swh_norm)
        mwd = self.mwd_normalizer.inverse_transform(mwd_norm)
        mwp = self.mwp_scaler.inverse_transform(mwp_norm)
        
        result = np.column_stack([swh.flatten(), mwd, mwp.flatten()])
        return result.reshape(shape[:-1] + (3,))

class CircularLoss(nn.Module):
    """Loss function with circular handling for MWD"""
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # predictions/targets: (B, 4, H, W)
        pred_swh = predictions[:, 0]
        pred_mwd_cos = predictions[:, 1]
        pred_mwd_sin = predictions[:, 2]
        pred_mwp = predictions[:, 3]
        
        true_swh = targets[:, 0]
        true_mwd_cos = targets[:, 1]
        true_mwd_sin = targets[:, 2]
        true_mwp = targets[:, 3]
        
        # Standard MSE for SWH and MWP
        swh_loss = F.mse_loss(pred_swh, true_swh)
        mwp_loss = F.mse_loss(pred_mwp, true_mwp)
        
        # Circular loss for MWD
        mwd_cos_loss = F.mse_loss(pred_mwd_cos, true_mwd_cos)
        mwd_sin_loss = F.mse_loss(pred_mwd_sin, true_mwd_sin)
        mwd_loss = mwd_cos_loss + mwd_sin_loss
        
        total_loss = swh_loss + mwd_loss + mwp_loss
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_loss': mwd_loss,
            'mwp_loss': mwp_loss
        }

class EarthFormerTrainer:
    """Trainer for EarthFormer wave model"""
    
    def __init__(self, config: EarthFormerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
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
        
        print(f"🌍 EarthFormer Trainer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup dataset"""
        print("\n📊 Setting up EarthFormer dataset...")
        
        dataset = EarthFormerWaveDataset(
            data_path=self.config.data_path,
            config=self.config
        )
        
        # Fit normalizers
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
        print(f"   Grid size: {self.config.lat_size} × {self.config.lon_size}")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def _fit_normalizers(self, dataset):
        """Fit normalizers on sample data"""
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(50, len(dataset)), 5):
            sample = dataset[i]
            features = sample['input'].numpy()
            targets = sample['target'].numpy()
            
            sample_features.append(features.reshape(-1, features.shape[-1]))
            sample_targets.append(targets)
        
        all_features = np.vstack(sample_features)
        all_targets = np.stack(sample_targets)
        
        # Replace NaN before fitting
        all_features = np.nan_to_num(all_features, nan=0.0)
        all_targets = np.nan_to_num(all_targets, nan=0.0)
        
        self.feature_normalizer.fit(all_features)
        self.target_normalizer.fit(all_targets)
        
        print(f"   Normalizers fitted on {len(all_features)} samples")
    
    def train_epoch(self, model, optimizer, criterion):
        """Train one epoch"""
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)  # (B, T, H, W, C)
            targets = batch['target'].to(self.device)  # (B, H, W, 3)
            
            # Normalize inputs
            B, T, H, W, C = inputs.shape
            inputs_flat = inputs.reshape(-1, C).cpu().numpy()
            inputs_norm = self.feature_normalizer.transform(inputs_flat)
            inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
            inputs = inputs.reshape(B, T, H, W, C)
            
            # Normalize targets
            targets_norm = self.target_normalizer.transform_targets(targets.cpu().numpy())
            targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(inputs)  # (B, 4, H, W)
            
            # Permute targets to match predictions
            targets = targets.permute(0, 3, 1, 2)  # (B, 4, H, W)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}")
        
        return np.mean(epoch_losses) if epoch_losses else float('inf')
    
    def validate(self, model, criterion):
        """Validate model"""
        model.eval()
        val_losses = []
        val_losses_by_var = {'swh': [], 'mwd': [], 'mwp': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Normalize inputs
                B, T, H, W, C = inputs.shape
                inputs_flat = inputs.reshape(-1, C).cpu().numpy()
                inputs_norm = self.feature_normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.reshape(B, T, H, W, C)
                
                # Normalize targets
                targets_norm = self.target_normalizer.transform_targets(targets.cpu().numpy())
                targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                
                # Forward pass
                predictions = model(inputs)
                
                # Permute targets
                targets = targets.permute(0, 3, 1, 2)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                val_losses.append(loss_dict['total_loss'].item())
                
                for var in ['swh', 'mwd', 'mwp']:
                    if f'{var}_loss' in loss_dict:
                        val_losses_by_var[var].append(loss_dict[f'{var}_loss'].item())
        
        mean_total_loss = np.mean(val_losses)
        mean_var_losses = {
            var: np.mean(losses) if losses else 0.0 
            for var, losses in val_losses_by_var.items()
        }
        
        return mean_total_loss, mean_var_losses
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting EarthFormer training...")
        
        # Setup data
        train_dataset, val_dataset = self.setup_data()
        
        # Create model
        model = EarthFormer(self.config).to(self.device)
        print(f"\n✅ EarthFormer model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Patch size: {self.config.patch_size}")
        print(f"   Patch grid: {model.h_patches} × {model.w_patches}")
        
        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
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
        
        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, model, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'val_loss': val_loss,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'history': self.history,
            'model_type': 'earthformer'
        }
        
        if is_best:
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        
        torch.save(checkpoint, path)
        print(f"   💾 Saved: {path.name}")
    
    def save_final_model(self, model):
        """Save final model"""
        final_data = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'training_history': self.history,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'earthformer'
        }
        
        path = self.output_dir / "earthformer_wave_model_final.pt"
        torch.save(final_data, path)
        
        print(f"\n💾 Final model saved to: {path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main execution function"""
    print("🌍 EARTHFORMER GLOBAL WAVE MODEL - BENCHMARK")
    print("=" * 70)
    print("Transformer-based architecture for wave prediction")
    print("Matching experimental setup with GNN model")
    print("=" * 70)
    
    # Configuration
    config = EarthFormerConfig()
    
    print(f"\n📋 Configuration:")
    print(f"   Data: {config.data_path}")
    print(f"   Grid: {config.lat_size} × {config.lon_size}")
    print(f"   Patch size: {config.patch_size}")
    print(f"   Architecture: {config.depth} layers, {config.heads} heads")
    print(f"   Input features: {config.num_input_features}")
    print(f"   Sequence length: {config.sequence_length} timesteps")
    print(f"   Device: {config.device}")
    
    # Create trainer and start training
    trainer = EarthFormerTrainer(config)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 EarthFormer training complete!")
    print(f"   Results saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()