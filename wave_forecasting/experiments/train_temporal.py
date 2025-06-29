# experiments/train_temporal.py
"""Temporal wave prediction experiment"""

def run_temporal_experiment():
    """Run temporal wave prediction experiment"""
    
    # Configuration for temporal prediction
    config = ExperimentConfig(
        name="temporal_wave_prediction",
        data=DataConfig(),
        mesh=MeshConfig(refinement_level=4),
        model=ModelConfig(hidden_dim=128, num_spatial_layers=6, num_temporal_layers=4),
        training=TrainingConfig(num_epochs=30, learning_rate=0.0005, batch_size=4),
        sequence_length=4,  # 24 hours of input
        forecast_horizon=24  # Predict 24 hours ahead
    )
    
    print(f"🌊 Running temporal experiment: {config.name}")
    print(f"   Input: {config.sequence_length * config.data.time_step_hours}h sequence")
    print(f"   Forecast: {config.forecast_horizon}h ahead")
    
    # Setup (same as spatial)
    era5_manager = ERA5DataManager(config.data)
    gebco_manager = GEBCODataManager(config.data)
    
    era5_atmo, era5_waves = era5_manager.load_month_data(2023, 1)
    gebco_data = gebco_manager.load_bathymetry()
    
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, config.data)
    mesh = IcosahedralMesh(config.mesh)
    mesh_loader = MeshDataLoader(mesh, interpolator, config.data)
    
    # Create temporal dataset
    train_dataset = TemporalWaveDataset(mesh_loader, config, max_samples=200)
    
    def temporal_collate_fn(batch):
        input_sequences = torch.stack([item['input_sequence'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        return {'input_sequences': input_sequences, 'targets': targets}
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                             shuffle=True, collate_fn=temporal_collate_fn)
    
    # Setup temporal model
    model = TemporalWaveGNN(config.model)
    
    region_indices = mesh.filter_region(config.data.lat_bounds, config.data.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, config.mesh.max_edge_distance_km)
    
    # Train
    trainer = TemporalTrainer(model, config.training, edge_index, edge_attr)
    history = trainer.train(train_loader, checkpoint_dir=config.checkpoint_dir)
    
    # Visualize
    visualize_predictions(model.spatial_encoder, mesh_loader, edge_index, edge_attr,
                         save_path=f"{config.output_dir}/temporal_predictions.png")
    plot_training_history(history, save_path=f"{config.output_dir}/temporal_training_history.png")
    
    print(f"✅ Temporal experiment complete! Results saved to {config.output_dir}")
    
    return model, history

if __name__ == "__main__":
    # Run experiments
    spatial_model, spatial_history = run_spatial_experiment()
    temporal_model, temporal_history = run_temporal_experiment()
