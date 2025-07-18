# experiments/train_spatial.py
"""Spatial wave prediction experiment"""
import torch
from torch.utils.data import DataLoader

from config.base import DataConfig, ExperimentConfig, MeshConfig, ModelConfig, TrainingConfig
from data.datasets import MeshDataLoader, SpatialWaveDataset
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from experiments.download_data import ERA5Downloader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from models.spatial import SpatialWaveGNN
from training.trainers import SpatialTrainer
from utils.visualization import visualize_predictions, plot_training_history


def run_spatial_experiment():
    """Run spatial wave prediction experiment"""
    
    # Configuration
    config = ExperimentConfig(
        name="spatial_wave_prediction",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=4),
        model=ModelConfig(hidden_dim=128, num_spatial_layers=8),
        training=TrainingConfig(num_epochs=50, learning_rate=0.001)
    )
    
    print(f"🌊 Running experiment: {config.name}")
    
    # Setup data
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    # Load first month data (extend this for multi-month)
    era5_atmo, era5_waves = era5_manager.load_month_data(2023, 1)
    gebco_data = gebco_manager.load_bathymetry()
    
    # Setup interpolation and mesh
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh = IcosahedralMesh(config.mesh)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # Create datasets
    train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=50)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    
    # Setup model and training
    model = SpatialWaveGNN(config.model)
    
    # Compute edges for this region
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    # Train
    trainer = SpatialTrainer(model, config.training, edge_index, edge_attr)
    history = trainer.train(train_loader, checkpoint_dir=config.checkpoint_dir)
    
    # Visualize results
    visualize_predictions(model, mesh_loader, edge_index, edge_attr, 
                         save_path=f"{config.output_dir}/predictions.png")
    plot_training_history(history, save_path=f"{config.output_dir}/training_history.png")
    
    print(f"✅ Experiment complete! Results saved to {config.output_dir}")
    
    return model, history

def run_spatial_experiment_with_logging():
    """Enhanced spatial experiment with comprehensive logging"""
    
    # Configuration
    config = ExperimentConfig(
        name="spatial_wave_prediction_v2",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=4),
        model=ModelConfig(hidden_dim=128, num_spatial_layers=8),
        training=TrainingConfig(num_epochs=50, learning_rate=0.001)
    )
    
    print(f"🌊 Running logged experiment: {config.name}")
    
    # Setup data
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    # Load first month data (extend this for multi-month)
    era5_atmo, era5_waves = era5_manager.load_month_data(2023, 1)
    gebco_data = gebco_manager.load_bathymetry()
    
    # Setup interpolation and mesh
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh = IcosahedralMesh(config.mesh)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # Create datasets
    train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=50)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    
    # Setup model and training
    model = SpatialWaveGNN(config.model)
    
    # Compute edges for this region
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    
    # Create trainer WITH logging
    trainer = SpatialTrainer(
        model=model, 
        config=config.training, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        experiment_config=config  # ← This enables logging
    )
    
    # Add custom notes
    # trainer.logger.add_note("Testing new wave variable configuration")
    # trainer.logger.add_note(f"Using {len(region_indices)} mesh nodes")

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    
    # Setup model and training
    model = SpatialWaveGNN(config.model)
    
    # Compute edges for this region
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    # Train
    trainer = SpatialTrainer(model, config.training, edge_index, edge_attr)
    history = trainer.train(train_loader, checkpoint_dir=config.checkpoint_dir)
    
    # Visualize results
    visualize_predictions(model, mesh_loader, edge_index, edge_attr, 
                         save_path=f"{config.output_dir}/predictions.png")
    plot_training_history(history, save_path=f"{config.output_dir}/training_history.png")
    
    print(f"✅ Experiment complete! Results saved to {config.output_dir}")
    
    return model, history
    
    # Train with automatic logging
    history = trainer.train(train_loader, val_loader=None)
    
    # The logger automatically generates:
    # - logs/spatial_wave_prediction_v2_20241215_143022/config.json
    # - logs/spatial_wave_prediction_v2_20241215_143022/plots/training_summary.png  
    # - logs/spatial_wave_prediction_v2_20241215_143022/experiment_report.md
    # - logs/spatial_wave_prediction_v2_20241215_143022/checkpoints/
    
    return trainer.logger.experiment_id