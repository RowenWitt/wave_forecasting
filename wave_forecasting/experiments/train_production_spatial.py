"""Production-scale spatial wave prediction experiment"""

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from config.base import ExperimentConfig, DataConfig, MeshConfig, ModelConfig, TrainingConfig
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader, SpatialWaveDataset
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from models.spatial import SpatialWaveGNN
from training.trainers import SpatialTrainer
from utils.visualization import visualize_predictions, plot_training_history
from utils.metrics import evaluate_model_performance, calculate_wave_metrics

class MultiYearDataset:
    """Handles multiple years of ERA5 data efficiently"""
    
    def __init__(self, era5_manager: ERA5DataManager, gebco_manager: GEBCODataManager, 
                 mesh_loader: MeshDataLoader, years: list, months_per_year: int = 12):
        self.era5_manager = era5_manager
        self.gebco_manager = gebco_manager
        self.mesh_loader = mesh_loader
        self.years = years
        self.months_per_year = months_per_year
        
        # Calculate total timesteps
        self.total_timesteps = self._calculate_total_timesteps()
        print(f"📅 Multi-year dataset: {len(years)} years, ~{self.total_timesteps} timesteps")
    
    def _calculate_total_timesteps(self):
        """Estimate total timesteps across all years"""
        # Rough estimate: 124 timesteps per month (4 per day * 31 days)
        return len(self.years) * self.months_per_year * 124
    
    def create_samples(self, max_samples: int = None, train_split: float = 0.8):
        """Create training samples from multiple years"""
        
        print(f"🔄 Creating samples from {len(self.years)} years of data...")
        
        all_samples = []
        samples_per_year = max_samples // len(self.years) if max_samples else 1000
        
        for year in self.years:
            print(f"  Processing year {year}...")
            year_samples = []
            
            # Process each month in the year
            for month in range(1, self.months_per_year + 1):
                try:
                    # Load month data
                    era5_atmo, era5_waves = self.era5_manager.load_month_data(year, month)
                    gebco_data = self.gebco_manager.load_bathymetry()
                    
                    # Create interpolator for this month
                    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, self.mesh_loader.config)
                    
                    # Temporarily update mesh loader
                    old_interpolator = self.mesh_loader.interpolator
                    self.mesh_loader.interpolator = interpolator
                    
                    # Create samples for this month (limited timesteps to manage memory)
                    month_timesteps = min(50, len(era5_atmo.valid_time) - 1)
                    month_dataset = SpatialWaveDataset(self.mesh_loader, num_timesteps=month_timesteps)
                    year_samples.extend(month_dataset.samples)
                    
                    # Restore original interpolator
                    self.mesh_loader.interpolator = old_interpolator
                    
                    print(f"    Month {month:02d}: {len(month_dataset.samples)} samples")
                    
                except Exception as e:
                    print(f"    ⚠️  Skipping {year}-{month:02d}: {e}")
                    continue
            
            # Limit samples per year if specified
            if samples_per_year and len(year_samples) > samples_per_year:
                year_samples = year_samples[:samples_per_year]
            
            all_samples.extend(year_samples)
            print(f"  Year {year}: {len(year_samples)} total samples")
        
        print(f"✅ Created {len(all_samples)} total samples")
        
        # Split into train/validation
        train_size = int(train_split * len(all_samples))
        val_size = len(all_samples) - train_size
        
        # Convert to dataset format
        full_dataset = SimpleSampleDataset(all_samples)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"📊 Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        return train_dataset, val_dataset

class SimpleSampleDataset(torch.utils.data.Dataset):
    """Simple wrapper for pre-created samples"""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_features, target_waves = self.samples[idx]
        return {'features': input_features, 'targets': target_waves}

def run_production_spatial_experiment():
    """Production-scale spatial experiment with validation and analysis"""
    
    # Production configuration
    config = ExperimentConfig(
        name="production_spatial_multiyear_highres",
        data=DataConfig(),
        mesh=MeshConfig(
            refinement_level=5,  # ~40k nodes (~28km spacing)
            max_edge_distance_km=250.0  # Tighter connectivity
        ),
        model=ModelConfig(
            hidden_dim=256,  # Larger model
            num_spatial_layers=12,  # Deeper network
            edge_features=3,
            output_features=3
        ),
        training=TrainingConfig(
            batch_size=4,  # Smaller batch for larger model
            num_epochs=100,  # Longer training
            learning_rate=0.0005,  # Conservative LR
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            mse_weight=1.0,
            physics_weight=0.15,  # Slightly higher physics weight
            lr_decay_epochs=20,  # Decay every 20 epochs
            lr_decay_factor=0.7
        ),
        max_training_samples=20000  # 20k samples across all years
    )
    
    print(f"🚀 Production Spatial Experiment: {config.name}")
    print(f"   Mesh: {config.mesh.approx_node_count:,} nodes")
    print(f"   Model: {config.model.hidden_dim}d, {config.model.num_spatial_layers} layers")
    print(f"   Training: {config.training.num_epochs} epochs, {config.max_training_samples:,} samples")
    
    # Setup data managers
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    # Create high-resolution mesh
    print("🔧 Building high-resolution mesh...")
    mesh = IcosahedralMesh(config.mesh)
    
    # Create initial interpolator (we'll update this per year)
    era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)  # Initial data
    gebco_data = gebco_manager.load_bathymetry()
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # Create multi-year dataset
    print("📊 Creating multi-year dataset...")
    multiyear_data = MultiYearDataset(
        era5_manager, gebco_manager, mesh_loader, 
        years=[2020, 2021, 2022],  # 3 years for now
        months_per_year=12
    )
    
    train_dataset, val_dataset = multiyear_data.create_samples(
        max_samples=config.max_training_samples,
        train_split=0.8
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=2,  # Parallel loading
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup model
    print("🧠 Setting up production model...")
    model = SpatialWaveGNN(config.model)
    
    # Compute edges for high-resolution mesh
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    print(f"🔗 Graph connectivity: {len(region_indices)} nodes, {edge_index.shape[1]} edges")
    
    # Enhanced trainer with validation
    trainer = EnhancedSpatialTrainer(
        model=model, 
        config=config.training, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        experiment_config=config
    )
    
    # Add experiment notes
    trainer.logger.add_note(f"Production experiment with {len(region_indices)} mesh nodes")
    trainer.logger.add_note(f"Multi-year training: 2020-2022")
    trainer.logger.add_note(f"High-resolution mesh: level {config.mesh.refinement_level}")
    
    # Train with validation
    print("🚀 Starting production training...")
    history = trainer.train(train_loader, val_loader)
    
    # Comprehensive evaluation
    print("📊 Running comprehensive evaluation...")
    final_metrics = evaluate_production_model(model, val_loader, edge_index, edge_attr, config)
    
    # Save final model
    final_model_path = Path(config.checkpoint_dir) / f"{config.name}_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'region_indices': region_indices
    }, final_model_path)
    
    trainer.logger.add_note(f"Final model saved to {final_model_path}")
    
    # Generate comprehensive visualizations
    create_production_visualizations(model, mesh_loader, edge_index, edge_attr, config)
    
    print(f"✅ Production experiment complete!")
    print(f"   Experiment ID: {trainer.logger.experiment_id}")
    print(f"   Final validation loss: {final_metrics.get('val_total', 'N/A'):.4f}")
    print(f"   Best SWH RMSE: {final_metrics.get('swh_rmse', 'N/A'):.4f}m")
    
    return trainer.logger.experiment_id, final_metrics

class EnhancedSpatialTrainer(SpatialTrainer):
    """Enhanced trainer with better validation and early stopping"""
    
    def __init__(self, model, config, edge_index, edge_attr, experiment_config=None):
        super().__init__(model, config, edge_index, edge_attr, experiment_config)
        
        # Early stopping parameters
        self.patience = 15
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train(self, train_loader, val_loader=None, checkpoint_dir="checkpoints"):
        """Enhanced training with early stopping and better validation"""
        
        # Log training start
        if self.logger:
            model_info = {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_type': type(self.model).__name__,
                'edge_count': self.edge_index.shape[1],
                'node_count': len(torch.unique(self.edge_index)),
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset) if val_loader else 0
            }
            self.logger.log_training_start(model_info)
        
        print(f"🚀 Starting enhanced spatial training: {self.config.num_epochs} epochs")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Training samples: {len(train_loader.dataset):,}")
        print(f"   Validation samples: {len(val_loader.dataset):,}" if val_loader else "   No validation set")
        
        train_history = {'train_loss': [], 'val_loss': []}
        
        try:
            for epoch in range(self.config.num_epochs):
                # Train epoch
                train_metrics = self.train_epoch(train_loader)
                
                # Validation with detailed metrics
                val_metrics = {}
                if val_loader:
                    val_metrics = self.validate_detailed(val_loader)
                
                # Learning rate scheduling
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                
                # Early stopping check
                if val_loader and val_metrics:
                    val_loss = val_metrics['total']
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                        
                        if self.logger:
                            self.logger.add_note(f"New best validation loss: {val_loss:.6f} at epoch {epoch+1}")
                    else:
                        self.patience_counter += 1
                
                # Log metrics
                if self.logger:
                    self.logger.log_epoch(epoch + 1, train_metrics, current_lr)
                    if val_metrics:
                        self.logger.log_validation(epoch + 1, val_metrics)
                
                # Console output
                print(f"Epoch {epoch+1:3d}/{self.config.num_epochs}: "
                      f"Train={train_metrics['total']:.6f}, "
                      f"MSE={train_metrics['mse']:.6f}, "
                      f"Physics={train_metrics['physics']:.6f}")
                
                if val_metrics:
                    print(f"   Val={val_metrics['total']:.6f}, "
                          f"SWH_RMSE={val_metrics.get('swh_rmse', 0):.4f}m, "
                          f"Patience={self.patience_counter}/{self.patience}")
                
                # Checkpointing
                if (epoch + 1) % 20 == 0 and self.logger:
                    checkpoint_state = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict()
                    }
                    self.logger.save_checkpoint(checkpoint_state, epoch + 1, train_metrics)
                
                # Track history
                train_history['train_loss'].append(train_metrics['total'])
                if val_metrics:
                    train_history['val_loss'].append(val_metrics['total'])
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"⏹️  Early stopping at epoch {epoch+1} (patience exceeded)")
                    if self.best_model_state:
                        self.model.load_state_dict(self.best_model_state)
                        print(f"   Restored best model (val_loss={self.best_val_loss:.6f})")
                    break
            
            # Final metrics
            final_metrics = train_metrics.copy()
            if val_metrics:
                final_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
                final_metrics['best_val_loss'] = self.best_val_loss
            
            if self.logger:
                self.logger.log_training_complete(final_metrics)
                self.logger.generate_report()
            
            print("✅ Enhanced spatial training complete!")
            return train_history
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(e)
            raise e
    
    def validate_detailed(self, dataloader):
        """Detailed validation with wave-specific metrics"""
        self.model.eval()
        val_losses = {'total': [], 'mse': [], 'physics': []}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features']
                targets = batch['targets']
                
                # Process batch
                batch_predictions = []
                for i in range(features.shape[0]):
                    pred = self.model(features[i], self.edge_index, self.edge_attr)
                    batch_predictions.append(pred)
                
                predictions = torch.stack(batch_predictions)
                loss_dict = self.criterion(predictions, targets)
                
                # Track losses
                for key in val_losses:
                    val_losses[key].append(loss_dict[f'{key}_loss'].item())
                
                # Collect for detailed metrics
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())
        
        # Calculate basic losses
        val_metrics = {key: np.mean(values) for key, values in val_losses.items()}
        
        # Calculate detailed wave metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        wave_metrics = calculate_wave_metrics(all_predictions, all_targets)
        val_metrics.update(wave_metrics)
        
        return val_metrics

def evaluate_production_model(model, test_loader, edge_index, edge_attr, config):
    """Comprehensive evaluation of production model"""
    
    print("🔍 Running comprehensive model evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features = batch['features']
            targets = batch['targets']
            
            # Process batch
            batch_predictions = []
            for i in range(features.shape[0]):
                pred = model(features[i], edge_index, edge_attr)
                batch_predictions.append(pred)
            
            predictions = torch.stack(batch_predictions)
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            if batch_idx % 50 == 0:
                print(f"   Processed {batch_idx+1}/{len(test_loader)} batches")
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"📊 Evaluation complete: {all_predictions.shape[0]:,} samples")
    
    # Calculate comprehensive metrics
    metrics = calculate_wave_metrics(all_predictions, all_targets)
    
    # Add summary statistics
    pred_stats = {
        'pred_swh_mean': all_predictions[:, :, 0].mean().item(),
        'pred_swh_std': all_predictions[:, :, 0].std().item(),
        'pred_swh_max': all_predictions[:, :, 0].max().item(),
        'target_swh_mean': all_targets[:, :, 0].mean().item(),
        'target_swh_std': all_targets[:, :, 0].std().item(),
        'target_swh_max': all_targets[:, :, 0].max().item(),
    }
    
    metrics.update(pred_stats)
    
    # Print key results
    print("\n📈 FINAL MODEL PERFORMANCE:")
    print("=" * 50)
    print(f"Wave Height (SWH):")
    print(f"  RMSE: {metrics.get('swh_rmse', 0):.4f}m")
    print(f"  MAE:  {metrics.get('swh_mae', 0):.4f}m") 
    print(f"  Skill Score: {metrics.get('swh_skill', 0):.3f}")
    print(f"  Correlation: {metrics.get('swh_corr', 0):.3f}")
    print()
    print(f"Wave Direction (MWD):")
    print(f"  RMSE: {metrics.get('mwd_rmse', 0):.4f}°")
    print(f"  MAE:  {metrics.get('mwd_mae', 0):.4f}°")
    print()
    print(f"Wave Period (MWP):")
    print(f"  RMSE: {metrics.get('mwp_rmse', 0):.4f}s")
    print(f"  MAE:  {metrics.get('mwp_mae', 0):.4f}s")
    print("=" * 50)
    
    return metrics

def create_production_visualizations(model, mesh_loader, edge_index, edge_attr, config):
    """Create comprehensive visualizations for production model"""
    
    print("📊 Creating production visualizations...")
    
    plots_dir = Path(config.output_dir) / "production_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Model prediction visualization
    visualize_predictions(
        model, mesh_loader, edge_index, edge_attr, 
        time_idx=0, save_path=plots_dir / "model_predictions.png"
    )
    
    print(f"✅ Production visualizations saved to {plots_dir}")

def run_quick_production_test():
    """Quick test with production settings but smaller scale"""
    
    config = ExperimentConfig(
        name="quick_production_test",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=4, max_edge_distance_km=300.0),  # Smaller
        model=ModelConfig(hidden_dim=128, num_spatial_layers=6),  # Smaller
        training=TrainingConfig(num_epochs=20, batch_size=8),  # Faster
        max_training_samples=1000  # Much smaller
    )
    
    print(f"🧪 Quick Production Test: {config.name}")
    
    # Use only 2020 data
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    mesh = IcosahedralMesh(config.mesh)
    era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
    gebco_data = gebco_manager.load_bathymetry()
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # Create small dataset with validation split
    train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=50)
    
    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.training.batch_size, shuffle=False)
    
    # Model and training
    model = SpatialWaveGNN(config.model)
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    trainer = EnhancedSpatialTrainer(model, config.training, edge_index, edge_attr, config)
    trainer.logger.add_note("Quick production test with validation")
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Quick evaluation
    final_metrics = evaluate_production_model(model, val_loader, edge_index, edge_attr, config)
    
    print(f"✅ Quick test complete: {trainer.logger.experiment_id}")
    return trainer.logger.experiment_id

if __name__ == "__main__":
    # Run quick test first
    print("🧪 Running quick production test...")
    quick_id = run_quick_production_test()
    
    print("\n" + "="*60)
    print("Ready to run full production experiment!")
    print("Uncomment the line below to start:")
    print("# production_id, metrics = run_production_spatial_experiment()")