# main.py
"""Main entry point for experiments"""
import os

import torch

from config.base import DataConfig, ExperimentConfig, MeshConfig, ModelConfig, TrainingConfig
from experiments.download_data import ERA5Downloader
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from mesh.icosahedral import IcosahedralMesh
from data.datasets import MeshDataLoader, SpatialWaveDataset
from torch.utils.data import DataLoader
from models.spatial import SpatialWaveGNN
from mesh.connectivity import compute_regional_edges

def setup_experiment_directories():
    """Create necessary directories for experiments"""
    directories = [
        "data/era5",
        "data/gebco", 
        "data/processed",
        "outputs",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def download_all_data():
    """Download all required data"""
    
    config = DataConfig()
    
    # Download ERA5 data
    print("🌊 Downloading ERA5 data...")
    downloader = ERA5Downloader(config)
    downloader.download_multi_year(2023, 2023)  # Start with 2023
    
    # Download GEBCO data
    print("🏔️  Downloading GEBCO data...")
    download_gebco_data(config)
    
    print("✅ All data downloaded!")

def run_quick_test():
    """Run a quick test of the system"""
    
    print("🧪 Running quick system test...")
    
    # Small configuration for testing
    config = ExperimentConfig(
        name="quick_test",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=3),  # Smaller mesh
        model=ModelConfig(hidden_dim=32, num_spatial_layers=2),
        training=TrainingConfig(num_epochs=5, batch_size=2),  # Quick training
        max_training_samples=20  # Very small dataset
    )
    
    try:
        # Setup data (assuming 2023/01 data exists)
        era5_manager = ERA5DataManager(config.data)
        gebco_manager = GEBCODataManager(config.data)
        
        era5_atmo, era5_waves = era5_manager.load_month_data(2023, 1)
        gebco_data = gebco_manager.load_bathymetry()
        
        # Quick spatial test
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
        mesh = IcosahedralMesh(config.mesh)
        mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
        
        # Create small dataset
        train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=10)
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size)
        
        # Quick model test
        model = SpatialWaveGNN(config.model)
        region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices)
        
        # Test forward pass
        sample_batch = next(iter(train_loader))
        sample_features = sample_batch['features'][0]
        
        with torch.no_grad():
            predictions = model(sample_features, edge_index, edge_attr)
        
        print(f"✅ Quick test passed!")
        print(f"   Model input: {sample_features.shape}")
        print(f"   Model output: {predictions.shape}")
        print(f"   Prediction ranges: SWH={predictions[:, 0].min():.3f}-{predictions[:, 0].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🌊 WAVE FORECASTING SYSTEM")
    print("=" * 50)
    
    # Setup
    setup_experiment_directories()
    
    # Check if we have data
    config = DataConfig()
    era5_manager = ERA5DataManager(config)
    
    if not era5_manager.available_files['atmospheric']:
        print("📥 No ERA5 data found. Starting download...")
        download_all_data()
    else:
        print("✅ ERA5 data found!")
        start_date, end_date = era5_manager.get_time_range()
        print(f"   Available data: {start_date} to {end_date}")
    
    # Run quick test
    if run_quick_test():
        print("\n🚀 System ready for full experiments!")
        print("\nNext steps:")
        print("1. Run: python -c 'from experiments.train_spatial import run_spatial_experiment; run_spatial_experiment()'")
        print("2. Run: python -c 'from experiments.train_temporal import run_temporal_experiment; run_temporal_experiment()'")
        print("3. Scale up with more data and larger models")
    else:
        print("\n❌ System test failed. Check your data and configuration.")

if __name__ == "__main__":
    main()
