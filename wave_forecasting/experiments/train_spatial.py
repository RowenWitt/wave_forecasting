# experiments/train_spatial.py
"""Spatial wave prediction experiment"""

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
