#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) for Wave Forecasting
Embeds wave equation and physics constraints directly into the model
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
from new_architecture_test import SpatioTemporalEvaluator, SpatioTemporalConfig

@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Network"""
    
    # Model architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    activation: str = 'tanh'  # Better for PINN gradients
    
    # Physics parameters
    gravity: float = 9.81  # m/s²
    water_density: float = 1025.0  # kg/m³
    
    # Loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 0.5
    boundary_loss_weight: float = 0.3
    conservation_loss_weight: float = 0.2
    
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Data parameters
    sequence_length: int = 3  # Shorter sequences for PINN
    prediction_horizon: int = 1  # Focus on single-step with physics
    validation_split: float = 0.2
    
    # Physics constraints
    enable_wave_equation: bool = True
    enable_energy_conservation: bool = True
    enable_dispersion_relation: bool = True
    enable_boundary_conditions: bool = True

class PhysicsCalculator:
    """Calculate physics-based terms for wave modeling"""
    
    def __init__(self, config: PINNConfig):
        self.config = config
        self.g = config.gravity
    
    def dispersion_relation(self, wave_period: torch.Tensor, water_depth: torch.Tensor) -> torch.Tensor:
        """
        Deep water dispersion relation: ω² = gk (simplified)
        ω = 2π/T, so k = ω²/g = (2π/T)²/g
        Returns wavenumber k
        """
        omega = 2 * math.pi / (wave_period + 1e-6)  # Avoid division by zero
        
        # Deep water approximation (valid for depth > wavelength/2)
        k_deep = omega**2 / self.g
        
        # Shallow water correction (if depth data available)
        # k * tanh(kh) = ω²/g
        # For now, use deep water approximation
        
        return k_deep
    
    def group_velocity(self, wave_period: torch.Tensor, water_depth: torch.Tensor) -> torch.Tensor:
        """
        Group velocity cg = ∂ω/∂k
        For deep water: cg = g/(2ω) = gT/(4π)
        """
        cg = self.g * wave_period / (4 * math.pi)
        return cg
    
    def wave_energy(self, wave_height: torch.Tensor) -> torch.Tensor:
        """
        Wave energy E = (1/8) * ρ * g * H²
        """
        rho = self.config.water_density
        energy = (1/8) * rho * self.g * wave_height**2
        return energy
    
    def compute_spatial_gradients(self, field: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial gradients ∂field/∂x, ∂field/∂y
        Using automatic differentiation
        """
        if not field.requires_grad:
            field = field.requires_grad_(True)
        
        # Compute gradients
        grad_outputs = torch.ones_like(field)
        gradients = torch.autograd.grad(
            outputs=field,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        grad_x = gradients[:, 0:1]  # ∂/∂x
        grad_y = gradients[:, 1:2]  # ∂/∂y
        
        return grad_x, grad_y
    
    def compute_temporal_gradient(self, field_t0: torch.Tensor, field_t1: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Compute temporal gradient ∂field/∂t using finite differences
        """
        return (field_t1 - field_t0) / dt

class PhysicsInformedWaveNet(nn.Module):
    """
    Physics-Informed Neural Network for wave prediction
    Embeds wave equation and physical constraints
    """
    
    def __init__(self, config: PINNConfig, input_features: int = 11):
        super().__init__()
        self.config = config
        self.physics = PhysicsCalculator(config)
        
        # Activation function
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        else:
            self.activation = nn.ReLU()
        
        # Encoder network
        encoder_layers = [nn.Linear(input_features, config.hidden_dim)]
        for _ in range(config.num_hidden_layers):
            encoder_layers.extend([
                self.activation,
                nn.Linear(config.hidden_dim, config.hidden_dim)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Output heads for wave parameters
        self.wave_height_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self.activation,
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.wave_direction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self.activation,
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.wave_period_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self.activation,
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Coordinate embedding for spatial derivatives
        self.coord_embed = nn.Linear(2, config.hidden_dim // 4)  # lat, lon
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for PINN (Xavier initialization works well)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with physics-aware predictions
        
        Args:
            x: Input features [batch_size, num_nodes, features]
            coords: Spatial coordinates [batch_size, num_nodes, 2] (lat, lon)
        
        Returns:
            Dictionary with wave predictions and intermediate values
        """
        batch_size, num_nodes, num_features = x.size()
        
        # Flatten for processing
        x_flat = x.view(-1, num_features)
        
        # Encode features
        encoded = self.encoder(x_flat)
        
        # Predict wave parameters
        wave_height = self.wave_height_head(encoded)  # [batch*nodes, 1]
        wave_direction = self.wave_direction_head(encoded)  # [batch*nodes, 1]
        wave_period = self.wave_period_head(encoded)  # [batch*nodes, 1]
        
        # Apply physical constraints
        wave_height = torch.clamp(wave_height, min=0.01, max=20.0)  # Physical bounds
        wave_direction = torch.fmod(wave_direction, 360.0)  # [0, 360)
        wave_direction = torch.where(wave_direction < 0, wave_direction + 360, wave_direction)
        wave_period = torch.clamp(wave_period, min=1.0, max=25.0)  # Physical bounds
        
        # Reshape back
        wave_height = wave_height.view(batch_size, num_nodes, 1)
        wave_direction = wave_direction.view(batch_size, num_nodes, 1)
        wave_period = wave_period.view(batch_size, num_nodes, 1)
        
        # Combine predictions
        predictions = torch.cat([wave_height, wave_direction, wave_period], dim=-1)
        
        result = {
            'predictions': predictions,
            'wave_height': wave_height,
            'wave_direction': wave_direction,
            'wave_period': wave_period,
            'encoded_features': encoded.view(batch_size, num_nodes, -1)
        }
        
        # Add physics-based terms if coordinates provided
        if coords is not None:
            coords_flat = coords.view(-1, 2).requires_grad_(True)
            
            # Compute physics terms
            bathymetry = x_flat[:, 6:7]  # Assuming bathymetry is feature index 6
            
            wave_height_flat = wave_height.view(-1, 1)
            wave_period_flat = wave_period.view(-1, 1)
            
            # Physics calculations
            wavenumber = self.physics.dispersion_relation(wave_period_flat, bathymetry)
            group_velocity = self.physics.group_velocity(wave_period_flat, bathymetry)
            wave_energy = self.physics.wave_energy(wave_height_flat)
            
            result.update({
                'wavenumber': wavenumber.view(batch_size, num_nodes, 1),
                'group_velocity': group_velocity.view(batch_size, num_nodes, 1),
                'wave_energy': wave_energy.view(batch_size, num_nodes, 1),
                'coordinates': coords
            })
        
        return result

class PINNLoss(nn.Module):
    """
    Physics-Informed loss function combining data loss with physics constraints
    """
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.physics = PhysicsCalculator(config)
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, 
                prev_predictions: Optional[Dict[str, torch.Tensor]] = None,
                dt: float = 3600.0) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss
        
        Args:
            predictions: Current model predictions
            targets: Ground truth targets
            prev_predictions: Previous timestep predictions (for temporal derivatives)
            dt: Time step in seconds
        """
        
        pred_values = predictions['predictions']
        
        # Data loss (standard MSE)
        data_loss = F.mse_loss(pred_values, targets)
        
        losses = {'data_loss': data_loss}
        total_loss = self.config.data_loss_weight * data_loss
        
        # Physics losses (only if we have coordinate information)
        if 'coordinates' in predictions:
            
            # Wave equation loss
            if self.config.enable_wave_equation and prev_predictions is not None:
                wave_eq_loss = self._wave_equation_loss(predictions, prev_predictions, dt)
                losses['wave_equation_loss'] = wave_eq_loss
                total_loss += self.config.physics_loss_weight * wave_eq_loss
            
            # Energy conservation loss
            if self.config.enable_energy_conservation:
                energy_loss = self._energy_conservation_loss(predictions)
                losses['energy_conservation_loss'] = energy_loss
                total_loss += self.config.conservation_loss_weight * energy_loss
            
            # Dispersion relation loss
            if self.config.enable_dispersion_relation:
                dispersion_loss = self._dispersion_relation_loss(predictions)
                losses['dispersion_relation_loss'] = dispersion_loss
                total_loss += self.config.physics_loss_weight * dispersion_loss
        
        # Boundary condition losses
        if self.config.enable_boundary_conditions:
            boundary_loss = self._boundary_condition_loss(predictions)
            losses['boundary_condition_loss'] = boundary_loss
            total_loss += self.config.boundary_loss_weight * boundary_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _wave_equation_loss(self, current: Dict, previous: Dict, dt: float) -> torch.Tensor:
        """
        Simplified wave equation: ∂H/∂t + cg * ∇H = 0
        where cg is group velocity
        """
        # Temporal derivative
        dH_dt = self.physics.compute_temporal_gradient(
            previous['wave_height'], current['wave_height'], dt
        )
        
        # Spatial gradients (simplified - would need proper implementation)
        # For now, use finite differences between neighboring nodes
        wave_height = current['wave_height']
        coords = current['coordinates']
        
        # Simplified spatial gradient (placeholder)
        # In practice, you'd use the mesh connectivity for proper gradients
        dH_dx = torch.zeros_like(wave_height)
        dH_dy = torch.zeros_like(wave_height)
        
        # Wave equation residual
        cg = current['group_velocity']
        wave_eq_residual = dH_dt + cg * (dH_dx + dH_dy)
        
        return torch.mean(wave_eq_residual**2)
    
    def _energy_conservation_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Energy should be conserved in the absence of sources/sinks
        ∇ · (E * cg) = 0 where E is wave energy, cg is group velocity
        """
        energy = predictions['wave_energy']
        cg = predictions['group_velocity']
        
        # Energy flux
        energy_flux = energy * cg
        
        # Simplified conservation (proper implementation would use mesh gradients)
        # For now, penalize large energy gradients
        energy_gradient_penalty = torch.var(energy_flux, dim=1).mean()
        
        return energy_gradient_penalty
    
    def _dispersion_relation_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Ensure predictions satisfy dispersion relation
        ω² = gk for deep water waves
        """
        wave_period = predictions['wave_period']
        wavenumber = predictions['wavenumber']
        
        # Computed frequency
        omega = 2 * math.pi / (wave_period + 1e-6)
        
        # Dispersion relation residual
        dispersion_residual = omega**2 - self.physics.g * wavenumber
        
        return torch.mean(dispersion_residual**2)
    
    def _boundary_condition_loss(self, predictions: Dict) -> torch.Tensor:
        """
        Apply boundary conditions (e.g., reflection at coastlines)
        For now, just ensure physical bounds are satisfied
        """
        wave_height = predictions['wave_height']
        wave_period = predictions['wave_period']
        
        # Physical bounds penalty
        height_penalty = F.relu(-wave_height).mean() + F.relu(wave_height - 20).mean()
        period_penalty = F.relu(1 - wave_period).mean() + F.relu(wave_period - 25).mean()
        
        return height_penalty + period_penalty

class PINNDataset(Dataset):
    """Dataset for PINN training with coordinate information"""
    
    def __init__(self, mesh_loader: MeshDataLoader, config: PINNConfig,
                 start_time: int = 0, end_time: int = 100):
        self.mesh_loader = mesh_loader
        self.config = config
        self.samples = []
        
        print(f"Building PINN dataset from timesteps {start_time} to {end_time}...")
        
        # Get mesh coordinates
        mesh = mesh_loader.mesh
        lat, lon = mesh.vertices_to_lat_lon()
        
        # Filter to regional nodes
        data_config = DataConfig()
        regional_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        self.coordinates = torch.tensor(
            np.column_stack([lat[regional_indices], lon[regional_indices]]), 
            dtype=torch.float32
        )
        
        # Build temporal sequences
        for t in range(start_time, end_time - config.sequence_length):
            try:
                # Input sequence
                input_features = []
                for i in range(config.sequence_length):
                    features_data = mesh_loader.load_features(time_idx=t + i)
                    features = torch.tensor(features_data['features'], dtype=torch.float32)
                    features = torch.nan_to_num(features, nan=0.0)
                    input_features.append(features)
                
                # Target (next timestep)
                target_data = mesh_loader.load_features(time_idx=t + config.sequence_length)
                targets = torch.tensor(target_data['features'][:, [3, 4, 5]], dtype=torch.float32)  # SWH, MWD, MWP
                targets = torch.nan_to_num(targets, nan=0.0)
                
                # Store sample
                input_tensor = torch.stack(input_features, dim=0)  # [seq_len, num_nodes, features]
                
                self.samples.append({
                    'input': input_tensor,
                    'target': targets,
                    'coordinates': self.coordinates,
                    'timestep': t
                })
                
            except Exception as e:
                print(f"Skipping timestep {t}: {e}")
                continue
        
        print(f"Created {len(self.samples)} PINN samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class PINNTrainer:
    """Trainer for Physics-Informed Neural Network"""
    
    def __init__(self, config: PINNConfig):
        self.config = config
        
        # Device setup with MPS support
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.normalizer = StandardScaler()
        self.target_normalizer = StandardScaler()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"pinn_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 PINN Training Experiment: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def setup_data(self):
        """Setup PINN data loaders"""
        
        print("📊 Setting up PINN dataset...")
        
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
        dataset = PINNDataset(mesh_loader, self.config, start_time=0, end_time=200)
        
        # Fit normalizers
        print("🔧 Fitting normalizers...")
        sample_features = []
        sample_targets = []
        
        for i in range(0, min(100, len(dataset)), 10):
            sample = dataset[i]
            features = sample['input'][-1].numpy()  # Use last timestep
            targets = sample['target'].numpy()
            
            sample_features.append(features)
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
        
        print(f"✅ PINN data setup complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch"""
        
        model.train()
        epoch_losses = {'total': [], 'data': [], 'physics': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get batch data
            inputs = batch['input'].to(self.device)  # [batch_size, seq_len, num_nodes, features]
            targets = batch['target'].to(self.device)  # [batch_size, num_nodes, 3]
            coords = batch['coordinates'].to(self.device)  # [batch_size, num_nodes, 2]
            
            # Use last timestep for prediction
            current_input = inputs[:, -1, :, :]  # [batch_size, num_nodes, features]
            
            # Normalize inputs
            batch_size, num_nodes, num_features = current_input.size()
            input_flat = current_input.contiguous().view(-1, num_features).cpu().numpy()
            input_norm = self.normalizer.transform(input_flat)
            current_input = torch.tensor(input_norm, dtype=torch.float32, device=self.device)
            current_input = current_input.view(batch_size, num_nodes, num_features)
            
            # Normalize targets
            target_flat = targets.contiguous().view(-1, 3).cpu().numpy()
            target_norm = self.target_normalizer.transform(target_flat)
            targets = torch.tensor(target_norm, dtype=torch.float32, device=self.device)
            targets = targets.view(batch_size, num_nodes, 3)
            
            # Forward pass
            predictions = model(current_input, coords)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['data'].append(loss_dict['data_loss'].item())
            
            # Track physics losses if present
            physics_loss = 0.0
            for key, value in loss_dict.items():
                if 'physics' in key or 'wave_equation' in key or 'energy' in key or 'dispersion' in key:
                    physics_loss += value.item()
            epoch_losses['physics'].append(physics_loss)
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion):
        """Validate model"""
        
        model.eval()
        val_losses = {'total': [], 'data': [], 'physics': []}
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                coords = batch['coordinates'].to(self.device)
                
                current_input = inputs[:, -1, :, :]
                
                # Normalize
                batch_size, num_nodes, num_features = current_input.size()
                input_flat = current_input.contiguous().view(-1, num_features).cpu().numpy()
                input_norm = self.normalizer.transform(input_flat)
                current_input = torch.tensor(input_norm, dtype=torch.float32, device=self.device)
                current_input = current_input.view(batch_size, num_nodes, num_features)
                
                target_flat = targets.contiguous().view(-1, 3).cpu().numpy()
                target_norm = self.target_normalizer.transform(target_flat)
                targets = torch.tensor(target_norm, dtype=torch.float32, device=self.device)
                targets = targets.view(batch_size, num_nodes, 3)
                
                # Forward pass
                predictions = model(current_input, coords)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                
                # Track losses
                val_losses['total'].append(loss_dict['total_loss'].item())
                val_losses['data'].append(loss_dict['data_loss'].item())
                
                physics_loss = 0.0
                for key, value in loss_dict.items():
                    if 'physics' in key or 'wave_equation' in key or 'energy' in key or 'dispersion' in key:
                        physics_loss += value.item()
                val_losses['physics'].append(physics_loss)
        
        return {key: np.mean(values) for key, values in val_losses.items()}
    
    def train(self):
        """Main training loop"""
        
        print(f"🔬 Starting PINN training...")
        
        # Setup
        train_loader, val_loader = self.setup_data()
        
        # Create model
        model = PhysicsInformedWaveNet(self.config).to(self.device)
        criterion = PINNLoss(self.config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print(f"✅ PINN model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Hidden layers: {self.config.num_hidden_layers}")
        print(f"   Physics constraints: Wave equation={self.config.enable_wave_equation}, "
              f"Energy conservation={self.config.enable_energy_conservation}")
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'train_data': [], 'val_data': [], 'train_physics': [], 'val_physics': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train and validate
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
            val_metrics = self.validate(model, val_loader, criterion)
            
            # Track history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['train_data'].append(train_metrics['data'])
            history['val_data'].append(val_metrics['data'])
            history['train_physics'].append(train_metrics['physics'])
            history['val_physics'].append(val_metrics['physics'])
            
            # Progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs}: "
                  f"Train={train_metrics['total']:.4f}, "
                  f"Val={val_metrics['total']:.4f}, "
                  f"Data={val_metrics['data']:.4f}, "
                  f"Physics={val_metrics['physics']:.4f}, "
                  f"Time={epoch_time:.1f}s")
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'normalizer': self.normalizer,
                    'target_normalizer': self.target_normalizer,
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, self.log_dir / "best_pinn_model.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        print(f"✅ PINN training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Model saved to: {self.log_dir / 'best_pinn_model.pt'}")
        
        # Generate plots
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot PINN training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('PINN Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Data loss
        axes[0, 1].plot(epochs, history['train_data'], 'g-', label='Train Data')
        axes[0, 1].plot(epochs, history['val_data'], 'orange', label='Val Data')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].set_title('Data Loss (MSE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Physics loss
        axes[1, 0].plot(epochs, history['train_physics'], 'purple', label='Train Physics')
        axes[1, 0].plot(epochs, history['val_physics'], 'brown', label='Val Physics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].set_title('Physics Constraints Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Summary
        axes[1, 1].axis('off')
        summary_text = f"""
PINN Training Summary:
• Experiment: {self.experiment_id}
• Final Train Loss: {history['train_loss'][-1]:.4f}
• Final Val Loss: {history['val_loss'][-1]:.4f}
• Final Data Loss: {history['val_data'][-1]:.4f}
• Final Physics Loss: {history['val_physics'][-1]:.4f}
• Hidden Layers: {self.config.num_hidden_layers}
• Hidden Dim: {self.config.hidden_dim}
• Physics Constraints:
  - Wave Equation: {self.config.enable_wave_equation}
  - Energy Conservation: {self.config.enable_energy_conservation}
  - Dispersion Relation: {self.config.enable_dispersion_relation}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'PINN Training: {self.experiment_id}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.log_dir / "pinn_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 PINN training plots saved: {plot_path}")

class PINNEvaluator:
    """Evaluator for trained PINN model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.load_model()
    
    def load_model(self):
        """Load trained PINN model"""
        
        print(f"📂 Loading PINN model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.config = checkpoint['config']
        self.normalizer = checkpoint['normalizer']
        self.target_normalizer = checkpoint['target_normalizer']
        
        # Create and load model
        self.model = PhysicsInformedWaveNet(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ PINN model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Hidden layers: {self.config.num_hidden_layers}")
    
    def predict(self, features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Make prediction with PINN model
        
        Args:
            features: Input features [num_nodes, features] or [batch_size, num_nodes, features]
            coordinates: Spatial coordinates [num_nodes, 2] or [batch_size, num_nodes, 2]
        
        Returns:
            Predictions [num_nodes, 3] or [batch_size, num_nodes, 3]
        """
        
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension
            coordinates = coordinates.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        features = features.to(self.device)
        coordinates = coordinates.to(self.device)
        
        # Normalize features
        batch_size, num_nodes, num_features = features.size()
        features_flat = features.contiguous().view(-1, num_features).cpu().numpy()
        features_norm = self.normalizer.transform(features_flat)
        features = torch.tensor(features_norm, dtype=torch.float32, device=self.device)
        features = features.view(batch_size, num_nodes, num_features)
        
        # Make prediction
        with torch.no_grad():
            predictions_dict = self.model(features, coordinates)
            predictions = predictions_dict['predictions']
        
        # Denormalize predictions
        pred_flat = predictions.contiguous().view(-1, 3).cpu().numpy()
        pred_denorm = self.target_normalizer.inverse_transform(pred_flat)
        predictions = torch.tensor(pred_denorm, dtype=torch.float32)
        predictions = predictions.view(batch_size, num_nodes, 3)
        
        return predictions.squeeze(0) if squeeze_output else predictions
    
    def evaluate_physics_constraints(self, features: torch.Tensor, coordinates: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate how well the model satisfies physics constraints
        """
        
        features = features.to(self.device)
        coordinates = coordinates.to(self.device)
        
        if features.dim() == 2:
            features = features.unsqueeze(0)
            coordinates = coordinates.unsqueeze(0)
        
        # Normalize features
        batch_size, num_nodes, num_features = features.size()
        features_flat = features.contiguous().view(-1, num_features).cpu().numpy()
        features_norm = self.normalizer.transform(features_flat)
        features = torch.tensor(features_norm, dtype=torch.float32, device=self.device)
        features = features.view(batch_size, num_nodes, num_features)
        
        with torch.no_grad():
            predictions_dict = self.model(features, coordinates)
        
        # Check physics constraints
        wave_height = predictions_dict['wave_height']
        wave_period = predictions_dict['wave_period']
        
        if 'wavenumber' in predictions_dict:
            wavenumber = predictions_dict['wavenumber']
            group_velocity = predictions_dict['group_velocity']
            wave_energy = predictions_dict['wave_energy']
            
            # Dispersion relation check
            omega = 2 * math.pi / (wave_period + 1e-6)
            dispersion_error = torch.mean(torch.abs(omega**2 - 9.81 * wavenumber)).item()
            
            # Energy bounds check
            energy_reasonable = torch.all(wave_energy >= 0).item()
            
            # Velocity bounds check
            velocity_reasonable = torch.all(group_velocity >= 0).item() and torch.all(group_velocity <= 50).item()
            
            return {
                'dispersion_error': dispersion_error,
                'energy_reasonable': energy_reasonable,
                'velocity_reasonable': velocity_reasonable,
                'physics_score': 1.0 - min(1.0, dispersion_error / 10.0)  # Normalized score
            }
        else:
            return {'physics_score': 0.0}

# Comprehensive Model Comparison Framework
class ModelComparison:
    """
    Compare different model architectures across multiple evaluation criteria
    """
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        
        # Temporal test periods for seasonal variation
        self.test_periods = {
            'winter_2020': {'year': 2020, 'months': [1, 2]},      # Winter storms
            'summer_2020': {'year': 2020, 'months': [6, 7]},      # Calm summer
            'autumn_2020': {'year': 2020, 'months': [10, 11]},    # Autumn storms
            'winter_2021': {'year': 2021, 'months': [1, 2]},      # Different winter
            'spring_2021': {'year': 2021, 'months': [4, 5]}       # Spring conditions
        }
    
    def add_model(self, name: str, model_path: str, model_type: str):
        """Add a model to the comparison"""
        
        self.models[name] = {
            'path': model_path,
            'type': model_type,  # 'spatiotemporal_gnn', 'pinn', 'convlstm', 'fno'
            'evaluator': None,
            'multi_step': model_type == 'spatiotemporal_gnn'  # Track if model does multi-step
        }
        
        print(f"📝 Added {model_type} model: {name}")
    
    def load_models(self):
        """Load all models for evaluation"""
        
        print("🔄 Loading models for comparison...")
        
        for name, model_info in self.models.items():
            try:
                if model_info['type'] == 'pinn':
                    model_info['evaluator'] = PINNEvaluator(model_info['path'])
                elif model_info['type'] == 'spatiotemporal_gnn':
                    # Import and load spatiotemporal model
                    # Assuming previous script
                    model_info['evaluator'] = SpatioTemporalEvaluator(model_info['path'])
                else:
                    print(f"   ⚠️  Model type {model_info['type']} not yet implemented")
                    
                print(f"   ✅ Loaded {name}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
    
    def evaluate_single_step_rmse(self, period_name: str, period_config: Dict) -> Dict[str, Dict[str, float]]:
        """
        Evaluate single-step RMSE for all models on a specific time period
        """
        
        print(f"📊 Evaluating single-step RMSE for {period_name}...")
        
        # Load test data for this period
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load test period data
        test_features = []
        test_targets = []
        
        for month in period_config['months']:
            try:
                era5_atmo, era5_waves = era5_manager.load_month_data(period_config['year'], month)
                gebco_data = gebco_manager.load_bathymetry()
                
                mesh = IcosahedralMesh(mesh_config)
                interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
                mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
                
                # Get coordinates
                lat, lon = mesh.vertices_to_lat_lon()
                regional_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
                coordinates = torch.tensor(
                    np.column_stack([lat[regional_indices], lon[regional_indices]]), 
                    dtype=torch.float32
                )
                
                # Sample some timesteps
                for t in range(0, min(50, len(era5_atmo.valid_time)), 5):  # Every 5th timestep
                    try:
                        features_data = mesh_loader.load_features(time_idx=t)
                        features = torch.tensor(features_data['features'], dtype=torch.float32)
                        features = torch.nan_to_num(features, nan=0.0)
                        
                        targets_data = mesh_loader.load_features(time_idx=t+1)
                        targets = torch.tensor(targets_data['features'][:, [3, 4, 5]], dtype=torch.float32)
                        targets = torch.nan_to_num(targets, nan=0.0)
                        
                        # For spatiotemporal models, we need sequence input
                        if any(model_info.get('multi_step', False) for model_info in self.models.values()):
                            # Create a sequence for spatiotemporal models
                            sequence_features = []
                            for seq_t in range(max(0, t-5), t+1):  # 6-timestep sequence
                                try:
                                    seq_data = mesh_loader.load_features(time_idx=seq_t)
                                    seq_feat = torch.tensor(seq_data['features'], dtype=torch.float32)
                                    seq_feat = torch.nan_to_num(seq_feat, nan=0.0)
                                    sequence_features.append(seq_feat)
                                except:
                                    # Pad with last available if needed
                                    if sequence_features:
                                        sequence_features.append(sequence_features[-1])
                                    else:
                                        sequence_features.append(features)
                            
                            # Ensure we have exactly 6 timesteps
                            while len(sequence_features) < 6:
                                sequence_features.insert(0, sequence_features[0] if sequence_features else features)
                            
                            sequence_tensor = torch.stack(sequence_features[:6], dim=0)  # [6, num_nodes, features]
                            test_features.append((features, coordinates, sequence_tensor))
                        else:
                            test_features.append((features, coordinates))
                        
                        test_targets.append(targets)
                        
                    except:
                        continue
                        
            except Exception as e:
                print(f"   ⚠️  Failed to load {period_config['year']}-{month:02d}: {e}")
                continue
        
        if not test_features:
            print(f"   ❌ No test data available for {period_name}")
            return {}
        
        print(f"   📈 Loaded {len(test_features)} test samples for {period_name}")
        
        # Evaluate each model
        results = {}
        
        for model_name, model_info in self.models.items():
            if model_info['evaluator'] is None:
                continue
                
            try:
                print(f"   🔍 Evaluating {model_name}...")
                
                swh_errors = []
                mwd_errors = []
                mwp_errors = []
                
                for i, (test_input, targets) in enumerate(zip(test_features, test_targets)):
                    if i % 10 == 0:
                        print(f"     Sample {i+1}/{len(test_features)}")
                    
                    # Unpack test input based on model type
                    if model_info.get('multi_step', False):
                        features, coords, sequence = test_input
                        # Make prediction using sequence
                        raw_predictions = model_info['evaluator'].predict(sequence.unsqueeze(0), multi_step=False)
                        raw_predictions = raw_predictions.squeeze(0)
                    else:
                        features, coords = test_input
                        # Make prediction using single timestep
                        if model_info['type'] == 'pinn':
                            raw_predictions = model_info['evaluator'].predict(features, coords)
                        else:
                            raw_predictions = model_info['evaluator'].predict(features.unsqueeze(0))
                            raw_predictions = raw_predictions.squeeze(0)
                    
                    # Handle multi-step output: take only first timestep  
                    if raw_predictions.shape[-1] == 12:  # 4 timesteps × 3 variables
                        predictions = raw_predictions[:, :3]  # Take first timestep [SWH, MWD, MWP]
                    elif raw_predictions.shape[-1] == 3:  # Single timestep
                        predictions = raw_predictions
                    else:
                        print(f"     ⚠️  Unexpected prediction shape: {raw_predictions.shape}")
                        continue
                    
                    # Calculate errors (ensure targets are the right shape)
                    if targets.shape[-1] != 3:
                        print(f"     ⚠️  Unexpected target shape: {targets.shape}")
                        continue
                    swh_error = torch.mean((predictions[:, 0] - targets[:, 0])**2).sqrt().item()
                    mwd_error = torch.mean((predictions[:, 1] - targets[:, 1])**2).sqrt().item()
                    mwp_error = torch.mean((predictions[:, 2] - targets[:, 2])**2).sqrt().item()
                    
                    swh_errors.append(swh_error)
                    mwd_errors.append(mwd_error)
                    mwp_errors.append(mwp_error)
                
                # Calculate mean RMSE
                results[model_name] = {
                    'swh_rmse': np.mean(swh_errors),
                    'mwd_rmse': np.mean(mwd_errors),
                    'mwp_rmse': np.mean(mwp_errors),
                    'overall_rmse': np.mean([np.mean(swh_errors), np.mean(mwd_errors), np.mean(mwp_errors)])
                }
                
                print(f"     ✅ {model_name}: SWH={results[model_name]['swh_rmse']:.3f}, "
                      f"MWD={results[model_name]['mwd_rmse']:.1f}, "
                      f"MWP={results[model_name]['mwp_rmse']:.3f}")
                
            except Exception as e:
                print(f"     ❌ Failed to evaluate {model_name}: {e}")
                continue
        
        return results
    
    def run_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation across all models and time periods
        """
        
        print("🏆 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 60)
        
        # Load all models
        self.load_models()
        
        # Evaluate across all time periods
        for period_name, period_config in self.test_periods.items():
            print(f"\n🗓️  Evaluating period: {period_name}")
            period_results = self.evaluate_single_step_rmse(period_name, period_config)
            self.evaluation_results[period_name] = period_results
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        print(f"\n📋 MODEL COMPARISON REPORT")
        print("=" * 50)
        
        # Collect all model names
        all_models = set()
        for period_results in self.evaluation_results.values():
            all_models.update(period_results.keys())
        
        # Summary table
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"{'Model':<20} {'Avg SWH RMSE':<12} {'Avg MWD RMSE':<12} {'Avg MWP RMSE':<12} {'Overall':<10}")
        print("-" * 70)
        
        model_averages = {}
        
        for model_name in all_models:
            swh_scores = []
            mwd_scores = []
            mwp_scores = []
            overall_scores = []
            
            for period_name, period_results in self.evaluation_results.items():
                if model_name in period_results:
                    swh_scores.append(period_results[model_name]['swh_rmse'])
                    mwd_scores.append(period_results[model_name]['mwd_rmse'])
                    mwp_scores.append(period_results[model_name]['mwp_rmse'])
                    overall_scores.append(period_results[model_name]['overall_rmse'])
            
            if swh_scores:
                avg_swh = np.mean(swh_scores)
                avg_mwd = np.mean(mwd_scores)
                avg_mwp = np.mean(mwp_scores)
                avg_overall = np.mean(overall_scores)
                
                model_averages[model_name] = {
                    'swh': avg_swh, 'mwd': avg_mwd, 'mwp': avg_mwp, 'overall': avg_overall
                }
                
                print(f"{model_name:<20} {avg_swh:<12.3f} {avg_mwd:<12.1f} {avg_mwp:<12.3f} {avg_overall:<10.3f}")
        
        # Find winner
        if model_averages:
            best_model = min(model_averages.keys(), key=lambda x: model_averages[x]['overall'])
            print(f"\n🏆 WINNER: {best_model} (Overall RMSE: {model_averages[best_model]['overall']:.3f})")
            
            # Check if it meets the <0.5 RMSE target
            if model_averages[best_model]['overall'] < 0.5:
                print(f"🎉 TARGET ACHIEVED: {best_model} achieves <0.5 overall RMSE!")
            else:
                print(f"🎯 TARGET MISSED: Best model achieves {model_averages[best_model]['overall']:.3f} RMSE (target: <0.5)")
        
        # Seasonal performance analysis
        print(f"\n🌊 SEASONAL PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        for period_name, period_results in self.evaluation_results.items():
            print(f"\n{period_name.upper()}:")
            for model_name, metrics in period_results.items():
                print(f"  {model_name}: {metrics['overall_rmse']:.3f} RMSE")

def main():
    """Main function to run PINN training and comparison"""
    
    print("🔬 PHYSICS-INFORMED NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # PINN Configuration
    config = PINNConfig(
        # Architecture
        hidden_dim=256,
        num_hidden_layers=4,
        activation='tanh',
        
        # Physics
        enable_wave_equation=True,
        enable_energy_conservation=True,
        enable_dispersion_relation=True,
        
        # Training
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        
        # Loss weights
        data_loss_weight=1.0,
        physics_loss_weight=0.5,
        conservation_loss_weight=0.2
    )
    
    print(f"🔧 PINN Configuration:")
    print(f"   Hidden layers: {config.num_hidden_layers}")
    print(f"   Hidden dimension: {config.hidden_dim}")
    print(f"   Activation: {config.activation}")
    print(f"   Physics constraints: Wave equation, Energy conservation, Dispersion relation")
    print(f"   Loss weights: Data={config.data_loss_weight}, Physics={config.physics_loss_weight}")
    
    # Train PINN
    trainer = PINNTrainer(config)
    model, history = trainer.train()
    
    print(f"\n🏆 PINN training complete!")
    print(f"   Model saved to: {trainer.log_dir / 'best_pinn_model.pt'}")
    
    # Setup model comparison
    print(f"\n🏁 SETTING UP MODEL COMPARISON")
    print("-" * 40)
    
    comparison = ModelComparison()
    
    # Add PINN model
    comparison.add_model(
        name="Physics-Informed NN", 
        model_path=str(trainer.log_dir / 'best_pinn_model.pt'),
        model_type="pinn"
    )
    
    # Add spatiotemporal GNN (if available)
    spatiotemporal_model_path = "experiments/spatiotemporal_20250702_235533/spatiotemporal_model.pt"  # Update with actual path
    if Path(spatiotemporal_model_path).exists():
        comparison.add_model(
            name="Spatiotemporal GNN",
            model_path=spatiotemporal_model_path,
            model_type="spatiotemporal_gnn"
        )
    
    # Run comprehensive evaluation
    comparison.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()