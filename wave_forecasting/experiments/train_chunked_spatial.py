from data.datasets import ChunkedSpatialDataset
from training.trainers import ChunkedSpatialTrainer

def run_chunked_spatial_experiment():
    """
    NEW chunked experiment - existing experiments still work!
    """
    
    # Use EXISTING config classes
    from config.base import ExperimentConfig, DataConfig, MeshConfig, ModelConfig, TrainingConfig
    
    config = ExperimentConfig(
        name="chunked_spatial_production",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=5),  # High-res
        model=ModelConfig(hidden_dim=512, num_spatial_layers=16),  # Large model
        training=TrainingConfig(batch_size=16, num_epochs=50),
    )
    
    print(f"🧩 Chunked Spatial Experiment: {config.name}")
    
    # Use EXISTING data managers
    from data.loaders import ERA5DataManager, GEBCODataManager
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    # Use EXISTING mesh setup
    from mesh.icosahedral import IcosahedralMesh
    from mesh.connectivity import compute_regional_edges
    from data.datasets import MeshDataLoader
    from data.preprocessing import MultiResolutionInterpolator
    
    mesh = IcosahedralMesh(config.mesh)
    
    # Initial setup (same as existing)
    era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
    gebco_data = gebco_manager.load_bathymetry()
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # NEW: Create chunked dataset
    chunked_dataset = ChunkedSpatialDataset(
        era5_manager=era5_manager,
        gebco_manager=gebco_manager,
        mesh_loader=mesh_loader,
        years=[2020, 2021, 2022],  # 3 years
        chunk_size_months=6
    )
    
    # Create validation dataset (simple approach)
    val_samples = chunked_dataset.get_chunk_samples(0)[:200]  # Use first chunk for validation
    
    # Use EXISTING model setup
    from models.spatial import SpatialWaveGNN
    model = SpatialWaveGNN(config.model)
    
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    # NEW: Use chunked trainer
    trainer = ChunkedSpatialTrainer(
        model=model,
        config=config.training,
        edge_index=edge_index,
        edge_attr=edge_attr,
        experiment_config=config
    )
    
    trainer.logger.add_note("Chunked training with cached data")
    trainer.logger.add_note(f"Multi-year: {chunked_dataset.years}")
    
    # NEW: Chunked training
    print("🚀 Starting chunked training...")
    history = trainer.train_chunked(chunked_dataset, val_samples, samples_per_chunk=1000)
    
    print(f"✅ Chunked experiment complete: {trainer.logger.experiment_id}")
    return trainer.logger.experiment_id
