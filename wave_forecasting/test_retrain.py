#!/usr/bin/env python3
"""
Complete robust wave forecasting model training script
Fixes overfitting and brittleness issues from previous model
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

# Import existing components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader, SpatialWaveDataset
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from models.base import WaveMessageLayer
from config.base import DataConfig, MeshConfig

@dataclass
class RobustTrainingConfig:
    """Robust training configuration to prevent overfitting"""
    
    # Model architecture
    hidden_dim: int = 128          # Reduced from 256
    num_spatial_layers: int = 4    # Reduced from 12
    dropout: float = 0.3           # Added regularization
    use_batch_norm: bool = True    # Stabilize training
    
    # Training parameters
    num_epochs: int = 25           # Early stopping
    batch_size: int = 16           # Larger batches
    learning_rate: float = 1e-4    # Conservative LR
    weight_decay: float = 1e-2     # Strong regularization
    gradient_clip_norm: float = 0.5 # Gradient clipping
    
    # Regularization
    feature_dropout: float = 0.1   # Input feature dropout
    label_smoothing: float = 0.1   # Label smoothing
    early_stopping_patience: int = 5
    
    # Loss weights
    mse_weight: float = 0.7
    physics_weight: float = 0.2
    regularization_weight: float = 0.1
    
    # Data
    validation_split: float = 0.2
    max_training_samples: int = 5000  # Prevent overfitting on too much data
    normalize_features: bool = True
    normalize_targets: bool = True

class FeatureNormalizer:
    """Consistent feature normalization for training and evaluation"""
    
    def __init__(self):
        self.feature_scaler = RobustScaler()  # Robust to outliers
        self.target_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit scalers on training data"""
        # Remove NaN values for fitting
        valid_features = features[~np.isnan(features).any(axis=1)]
        valid_targets = targets[~np.isnan(targets).any(axis=1)]
        
        self.feature_scaler.fit(valid_features)
        self.target_scaler.fit(valid_targets)
        self.fitted = True
        
        print(f"📊 Normalizer fitted:")
        print(f"   Feature stats: mean={self.feature_scaler.center_[:3]}")
        print(f"   Target stats: mean={self.target_scaler.mean_}, std={self.target_scaler.scale_}")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        return self.feature_scaler.transform(features)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        return self.target_scaler.transform(targets)
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets back to original scale"""
        return self.target_scaler.inverse_transform(targets)

class ImprovedWaveLoss(nn.Module):
    """Improved loss function with physics constraints and regularization"""
    
    def __init__(self, config: RobustTrainingConfig):
        super().__init__()
        self.config = config
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                features: torch.Tensor = None, model: nn.Module = None) -> Dict[str, torch.Tensor]:
        """Compute improved loss with multiple components"""
        
        # Basic MSE loss with label smoothing
        if self.config.label_smoothing > 0:
            smoothed_targets = targets * (1 - self.config.label_smoothing) + \
                              torch.mean(targets) * self.config.label_smoothing
        else:
            smoothed_targets = targets
        
        mse_loss = F.mse_loss(predictions, smoothed_targets)
        
        # Physics constraints
        physics_loss = self._physics_constraints(predictions)
        
        # Regularization losses
        reg_loss = torch.tensor(0.0, device=predictions.device)
        
        if model is not None:
            # L2 regularization (handled by weight_decay in optimizer)
            pass
        
        # Total loss
        total_loss = (self.config.mse_weight * mse_loss + 
                     self.config.physics_weight * physics_loss + 
                     self.config.regularization_weight * reg_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss,
            'reg_loss': reg_loss
        }
    
    def _physics_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints to predictions"""
        
        # Wave height must be positive
        swh_penalty = F.relu(-predictions[:, 0]).mean()
        
        # Wave direction should be in [0, 360] - use soft constraints
        mwd_penalty = F.relu(predictions[:, 1] - 360).mean() + F.relu(-predictions[:, 1]).mean()
        
        # Wave period should be positive and reasonable [1, 20]
        mwp_penalty = F.relu(-predictions[:, 2]).mean() + F.relu(predictions[:, 2] - 20).mean()
        
        return swh_penalty + mwd_penalty + mwp_penalty

class RobustSpatialWaveGNN(nn.Module):
    """Robust spatial wave GNN with proper regularization"""
    
    def __init__(self, config: RobustTrainingConfig, input_features: int = 11):
        super().__init__()
        self.config = config
        
        # Feature dropout for regularization
        self.feature_dropout = nn.Dropout(config.feature_dropout)
        
        # Robust encoder with batch normalization
        encoder_layers = []
        
        # First layer: input -> hidden//2
        encoder_layers.extend([
            nn.Linear(input_features, config.hidden_dim // 2),
            nn.BatchNorm1d(config.hidden_dim // 2) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        ])
        
        # Second layer: hidden//2 -> hidden
        encoder_layers.extend([
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Fewer message passing layers with residual connections
        self.message_layers = nn.ModuleList([
            WaveMessageLayer(config.hidden_dim, 3, config.hidden_dim)  # edge_features=3
            for _ in range(config.num_spatial_layers)
        ])
        
        # Robust decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout * 0.5),  # Less dropout in decoder
            nn.Linear(config.hidden_dim // 2, 3)  # Output: [SWH, MWD, MWP]
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass with regularization"""
        
        # Apply feature dropout
        x = self.feature_dropout(x)
        
        # Encode
        h = self.encoder(x)
        
        # Message passing with residual connections
        for layer in self.message_layers:
            h_new = layer(h, edge_index, edge_attr)
            h = h + h_new  # Residual connection
        
        # Decode
        predictions = self.decoder(h)
        
        return predictions

class RobustDataset(torch.utils.data.Dataset):
    """Robust dataset with proper normalization and validation"""
    
    def __init__(self, spatial_dataset: SpatialWaveDataset, normalizer: FeatureNormalizer, 
                 config: RobustTrainingConfig):
        self.spatial_dataset = spatial_dataset
        self.normalizer = normalizer
        self.config = config
        
        # Pre-process and cache normalized samples
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare and normalize samples"""
        print(f"🔄 Preparing normalized dataset...")
        
        valid_samples = 0
        for i in range(len(self.spatial_dataset)):
            try:
                sample = self.spatial_dataset[i]
                features = sample['features'].numpy()
                targets = sample['targets'].numpy()
                
                # Check for NaN values
                if np.any(np.isnan(features)) or np.any(np.isnan(targets)):
                    continue
                
                # Normalize
                if self.config.normalize_features:
                    features = self.normalizer.transform_features(features)
                
                if self.config.normalize_targets:
                    targets = self.normalizer.transform_targets(targets)
                
                # Convert back to tensors
                features_tensor = torch.tensor(features, dtype=torch.float32)
                targets_tensor = torch.tensor(targets, dtype=torch.float32)
                
                self.samples.append({
                    'features': features_tensor,
                    'targets': targets_tensor
                })
                
                valid_samples += 1
                
                # Limit samples to prevent overfitting
                if valid_samples >= self.config.max_training_samples:
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Skipping sample {i}: {e}")
                continue
        
        print(f"✅ Prepared {len(self.samples)} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop"""
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class RobustTrainer:
    """Robust trainer with proper validation and monitoring"""
    
    def __init__(self, config: RobustTrainingConfig):
        self.config = config
        self.normalizer = FeatureNormalizer()
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"robust_spatial_{timestamp}"
        self.log_dir = Path("logs") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Robust Training Experiment: {self.experiment_id}")
    
    def setup_data(self):
        """Setup data with proper validation split"""
        
        print("📊 Setting up robust dataset...")
        
        # Load data
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Use 2020 data for training
        era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh and data loader
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create spatial dataset
        spatial_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=20)
        
        # Fit normalizer on a sample of the data
        print("🔧 Fitting normalizer...")
        sample_features = []
        sample_targets = []
        
        for i in range(min(100, len(spatial_dataset))):
            try:
                sample = spatial_dataset[i]
                features = sample['features'].numpy()
                targets = sample['targets'].numpy()
                
                if not (np.any(np.isnan(features)) or np.any(np.isnan(targets))):
                    sample_features.append(features)
                    sample_targets.append(targets)
            except:
                continue
        
        if sample_features:
            all_features = np.vstack(sample_features)
            all_targets = np.vstack(sample_targets)
            self.normalizer.fit(all_features, all_targets)
        else:
            raise ValueError("No valid samples found for normalization")
        
        # Create robust dataset
        robust_dataset = RobustDataset(spatial_dataset, self.normalizer, self.config)
        
        # Train/validation split
        val_size = int(self.config.validation_split * len(robust_dataset))
        train_size = len(robust_dataset) - val_size
        
        train_dataset, val_dataset = random_split(robust_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"✅ Data setup complete:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Setup graph connectivity
        region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
        
        return train_loader, val_loader, edge_index, edge_attr
    
    def create_model(self, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Create robust model with proper initialization"""
        
        print("🧠 Creating robust model...")
        
        model = RobustSpatialWaveGNN(self.config, input_features=11)
        
        # Loss function
        criterion = ImprovedWaveLoss(self.config)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        print(f"✅ Model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Hidden dim: {self.config.hidden_dim}")
        print(f"   Spatial layers: {self.config.num_spatial_layers}")
        
        return model, criterion, optimizer, scheduler, early_stopping, edge_index, edge_attr
    
    def train_epoch(self, model, train_loader, criterion, optimizer, edge_index, edge_attr):
        """Train for one epoch"""
        
        model.train()
        epoch_losses = {'total': [], 'mse': [], 'physics': [], 'reg': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            features = batch['features']
            targets = batch['targets']
            
            # Forward pass for each sample in batch
            batch_predictions = []
            for i in range(features.shape[0]):
                pred = model(features[i], edge_index, edge_attr)
                batch_predictions.append(pred)
            
            predictions = torch.stack(batch_predictions)
            
            # Compute loss
            loss_dict = criterion(predictions, targets, features=features, model=model)
            
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
    
    def validate(self, model, val_loader, criterion, edge_index, edge_attr):
        """Validate model"""
        
        model.eval()
        val_losses = {'total': [], 'mse': [], 'physics': [], 'reg': []}
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                targets = batch['targets']
                
                # Forward pass
                batch_predictions = []
                for i in range(features.shape[0]):
                    pred = model(features[i], edge_index, edge_attr)
                    batch_predictions.append(pred)
                
                predictions = torch.stack(batch_predictions)
                
                # Compute loss
                loss_dict = criterion(predictions, targets)
                
                # Track losses
                for key in val_losses:
                    if f'{key}_loss' in loss_dict:
                        val_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in val_losses.items()}
    
    def check_feature_sensitivity(self, model, val_loader, edge_index, edge_attr):
        """Check if model maintains feature sensitivity"""
        
        model.eval()
        
        with torch.no_grad():
            # Get a validation batch
            batch = next(iter(val_loader))
            features = batch['features'][0]  # First sample
            
            # Baseline prediction
            baseline_pred = model(features, edge_index, edge_attr)
            
            # Test sensitivity to each feature
            sensitivities = []
            for feat_idx in range(features.shape[1]):
                modified_features = features.clone()
                modified_features[:, feat_idx] += 1.0  # Add 1 standard deviation
                
                modified_pred = model(modified_features, edge_index, edge_attr)
                sensitivity = torch.mean(torch.abs(modified_pred - baseline_pred))
                sensitivities.append(sensitivity.item())
            
            avg_sensitivity = np.mean(sensitivities)
            max_sensitivity = np.max(sensitivities)
            
            return avg_sensitivity, max_sensitivity
    
    def train(self):
        """Main training loop"""
        
        print(f"🚀 Starting robust training...")
        
        # Setup
        train_loader, val_loader, edge_index, edge_attr = self.setup_data()
        model, criterion, optimizer, scheduler, early_stopping, edge_index, edge_attr = self.create_model(edge_index, edge_attr)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mse': [], 'val_mse': [],
            'feature_sensitivity': []
        }
        
        print(f"\n📈 Training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, edge_index, edge_attr)
            
            # Validate
            val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr)
            
            # Check feature sensitivity
            avg_sens, max_sens = self.check_feature_sensitivity(model, val_loader, edge_index, edge_attr)
            
            # Update learning rate
            scheduler.step()
            
            # Track history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['train_mse'].append(train_metrics['mse'])
            history['val_mse'].append(val_metrics['mse'])
            history['feature_sensitivity'].append(avg_sens)
            
            # Progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:2d}/{self.config.num_epochs}: "
                  f"Train={train_metrics['total']:.6f}, "
                  f"Val={val_metrics['total']:.6f}, "
                  f"Sens={avg_sens:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}, "
                  f"Time={epoch_time:.1f}s")
            
            # Early stopping
            if early_stopping(val_metrics['total'], model):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                model.load_state_dict(early_stopping.best_model_state)
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics)
        
        # Final evaluation
        print(f"\n📊 Final evaluation...")
        final_train_metrics = self.validate(model, train_loader, criterion, edge_index, edge_attr)
        final_val_metrics = self.validate(model, val_loader, criterion, edge_index, edge_attr)
        final_sens_avg, final_sens_max = self.check_feature_sensitivity(model, val_loader, edge_index, edge_attr)
        
        print(f"✅ Training complete!")
        print(f"   Final train loss: {final_train_metrics['total']:.6f}")
        print(f"   Final val loss: {final_val_metrics['total']:.6f}")
        print(f"   Feature sensitivity: {final_sens_avg:.6f} (avg), {final_sens_max:.6f} (max)")
        
        # Save final model
        self.save_final_model(model, history, final_train_metrics, final_val_metrics)
        
        # Generate plots
        self.plot_training_history(history)
        
        return model, history
    
    def save_checkpoint(self, model, optimizer, epoch, train_metrics, val_metrics):
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'normalizer_state': {
                'feature_scaler': self.normalizer.feature_scaler,
                'target_scaler': self.normalizer.target_scaler
            }
        }
        
        checkpoint_path = self.log_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def save_final_model(self, model, history, train_metrics, val_metrics):
        """Save final model with all metadata"""
        
        final_model = {
            'experiment_id': self.experiment_id,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'normalizer_state': {
                'feature_scaler': self.normalizer.feature_scaler,
                'target_scaler': self.normalizer.target_scaler
            },
            'training_history': history,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.log_dir / "final_robust_model.pt"
        torch.save(final_model, model_path)
        
        # Also save in checkpoints directory for compatibility
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(final_model, checkpoint_dir / f"robust_spatial_{self.experiment_id}.pt")
        
        print(f"💾 Final model saved: {model_path}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
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
        
        # Summary stats
        axes[1, 1].axis('off')
        summary_text = f"""
Training Summary:
• Experiment: {self.experiment_id}
• Final Train Loss: {history['train_loss'][-1]:.6f}
• Final Val Loss: {history['val_loss'][-1]:.6f}
• Final Sensitivity: {history['feature_sensitivity'][-1]:.6f}
• Hidden Dim: {self.config.hidden_dim}
• Spatial Layers: {self.config.num_spatial_layers}
• Dropout: {self.config.dropout}
• Learning Rate: {self.config.learning_rate}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Robust Model Training: {self.experiment_id}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.log_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training plots saved: {plot_path}")

def main():
    """Main training function"""
    
    print("🌊 ROBUST WAVE FORECASTING MODEL TRAINING")
    print("=" * 60)
    
    # Configuration
    config = RobustTrainingConfig(
        # Model architecture - conservative settings
        hidden_dim=128,
        num_spatial_layers=4,
        dropout=0.3,
        use_batch_norm=True,
        
        # Training - robust settings
        num_epochs=25,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-2,
        
        # Regularization
        early_stopping_patience=5,
        max_training_samples=3000,  # Prevent overfitting
        
        # Data processing
        normalize_features=True,
        normalize_targets=True
    )
    
    print(f"🔧 Training configuration:")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Spatial layers: {config.num_spatial_layers}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Weight decay: {config.weight_decay}")
    # print(f"   Max samples/

    robust_trainer = RobustTrainer(config)
    print('training')
    model, history =  robust_trainer.train()


if __name__ == '__main__':
    main()