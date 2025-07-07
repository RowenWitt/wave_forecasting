#!/usr/bin/env python3
"""
Diagnostic for the clean system performance issues
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def diagnose_prediction_issues():
    """Diagnose why predictions are static and RMSE is high"""
    
    print("🔍 DIAGNOSING PREDICTION ISSUES")
    print("=" * 40)
    
    # Load the clean system
    from prediction.forecasting import WavePredictor, AutoregressiveForecaster, ForecastConfig
    
    # Setup data
    from data.loaders import ERA5DataManager, GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
    # Find latest checkpoint
    logs_path = Path("logs")
    all_checkpoints = []
    for exp_dir in logs_path.iterdir():
        if exp_dir.is_dir():
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("*.pt"):
                    all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint)))
    
    all_checkpoints.sort(reverse=True)
    latest_checkpoint = all_checkpoints[0][1]
    
    # Load predictor
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
    # Setup data environment
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    era5_manager = ERA5DataManager(data_config)
    gebco_manager = GEBCODataManager(data_config)
    
    first_file = era5_manager.available_files['atmospheric'][0]
    filename = Path(first_file).stem
    year_month = filename.split('_')[-1]
    year, month = int(year_month[:4]), int(year_month[4:6])
    
    era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
    gebco_data = gebco_manager.load_bathymetry()
    
    mesh = IcosahedralMesh(mesh_config)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
    mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
    
    forecaster = AutoregressiveForecaster(predictor, mesh_loader)
    
    print(f"✅ System loaded successfully")
    
    # DIAGNOSTIC 1: Check initial vs ground truth features
    print(f"\n🔍 DIAGNOSTIC 1: Feature Analysis")
    print("-" * 30)
    
    initial_data = mesh_loader.load_features(time_idx=20)
    initial_features = torch.tensor(initial_data['features'], dtype=torch.float32)
    
    # Load ground truth for t+6h
    target_data = mesh_loader.load_features(time_idx=21)  # 20 + 1 step
    target_features = torch.tensor(target_data['features'], dtype=torch.float32)
    
    print(f"Initial features (t=20): {initial_features.shape}")
    print(f"Target features (t=21):  {target_features.shape}")
    print(f"Feature names: {predictor.feature_names}")
    print(f"Wave indices: {predictor.wave_indices}")
    
    # Compare wave variables specifically
    for i, wave_idx in enumerate(predictor.wave_indices):
        if wave_idx < initial_features.shape[1]:
            wave_name = predictor.feature_names[wave_idx]
            initial_vals = initial_features[:, wave_idx]
            target_vals = target_features[:, wave_idx]
            
            print(f"   {wave_name}:")
            print(f"     Initial: {initial_vals.min():.3f} to {initial_vals.max():.3f} (μ={initial_vals.mean():.3f})")
            print(f"     Target:  {target_vals.min():.3f} to {target_vals.max():.3f} (μ={target_vals.mean():.3f})")
    
    # DIAGNOSTIC 2: Check model prediction vs ground truth
    print(f"\n🔍 DIAGNOSTIC 2: Model Prediction Analysis")
    print("-" * 40)
    
    from data.preprocessing import clean_features_for_training
    clean_initial = clean_features_for_training(initial_features)
    
    # Make single prediction
    with torch.no_grad():
        prediction = predictor.predict(clean_initial)
    
    print(f"Model prediction shape: {prediction.shape}")
    print(f"Expected: [num_nodes, 3] for [SWH, MWD, MWP]")
    
    # Compare prediction to ground truth waves
    if len(predictor.wave_indices) >= 3:
        true_waves = target_features[:, predictor.wave_indices[:3]]
    else:
        true_waves = target_features[:, -3:]  # Fallback
    
    print(f"Ground truth waves shape: {true_waves.shape}")
    
    for i, wave_name in enumerate(['SWH', 'MWD', 'MWP']):
        if i < prediction.shape[1] and i < true_waves.shape[1]:
            pred_vals = prediction[:, i]
            true_vals = true_waves[:, i]
            
            rmse = torch.sqrt(torch.mean((pred_vals - true_vals)**2))
            bias = torch.mean(pred_vals - true_vals)
            
            print(f"   {wave_name}:")
            print(f"     Predicted: {pred_vals.min():.3f} to {pred_vals.max():.3f} (μ={pred_vals.mean():.3f})")
            print(f"     True:      {true_vals.min():.3f} to {true_vals.max():.3f} (μ={true_vals.mean():.3f})")
            print(f"     RMSE: {rmse:.4f}, Bias: {bias:+.3f}")
    
    # DIAGNOSTIC 3: Check autoregressive state updates
    print(f"\n🔍 DIAGNOSTIC 3: Autoregressive State Update Analysis")
    print("-" * 50)
    
    current_state = clean_initial.clone()
    
    print(f"Testing autoregressive updates:")
    for step in range(3):
        print(f"\n   Step {step}:")
        
        # Show relevant features before prediction
        if len(predictor.wave_indices) >= 3:
            wave_features = current_state[:, predictor.wave_indices]
            print(f"     Input wave features: {[f'{v:.3f}' for v in wave_features[0, :3]]}")
        
        # Make prediction
        with torch.no_grad():
            wave_pred = predictor.predict(current_state)
        
        print(f"     Prediction: {[f'{v:.3f}' for v in wave_pred[0, :3]]}")
        
        # Update state (same logic as forecaster)
        new_state = current_state.clone()
        if len(predictor.wave_indices) >= 3 and wave_pred.shape[1] >= 3:
            for i, wave_idx in enumerate(predictor.wave_indices[:3]):
                if i < wave_pred.shape[1]:
                    new_state[:, wave_idx] = wave_pred[:, i]
        
        # Check if state actually changed
        state_diff = torch.mean(torch.abs(new_state - current_state))
        print(f"     State change: {state_diff:.6f}")
        
        current_state = new_state
        
        # Check if this affects next prediction
        if step < 2:
            with torch.no_grad():
                next_pred = predictor.predict(current_state)
            
            pred_diff = torch.mean(torch.abs(next_pred - wave_pred))
            print(f"     Prediction change: {pred_diff:.6f}")
    
    # DIAGNOSTIC 4: Check if we're using the right evaluation method
    print(f"\n🔍 DIAGNOSTIC 4: Evaluation Method Check")
    print("-" * 35)
    
    # Compare our evaluation to training dataset evaluation
    try:
        from data.datasets import SpatialWaveDataset
        
        # Create training-style dataset
        train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=5)
        sample = train_dataset[0]
        
        train_input = sample['features']
        train_target = sample['targets']
        
        print(f"Training dataset format:")
        print(f"   Input shape: {train_input.shape}")
        print(f"   Target shape: {train_target.shape}")
        
        # Test model on training format
        with torch.no_grad():
            train_pred = predictor.predict(train_input)
        
        print(f"   Model output shape: {train_pred.shape}")
        
        # Calculate RMSE in training format
        if train_pred.shape == train_target.shape:
            train_rmse = torch.sqrt(torch.mean((train_pred - train_target)**2))
            train_bias = torch.mean(train_pred - train_target)
            
            print(f"   Training format RMSE: {train_rmse:.4f}")
            print(f"   Training format Bias: {train_bias:+.3f}")
            
            # This should be closer to your 0.21m performance!
        
    except Exception as e:
        print(f"   ❌ Training dataset test failed: {e}")
    
    return predictor, mesh_loader

def test_different_evaluation_strategies(predictor, mesh_loader):
    """Test different ways of evaluating to find the correct method"""
    
    print(f"\n🧪 TESTING DIFFERENT EVALUATION STRATEGIES")
    print("=" * 45)
    
    # Load data
    initial_data = mesh_loader.load_features(time_idx=20)
    target_data = mesh_loader.load_features(time_idx=21)
    
    initial_features = torch.tensor(initial_data['features'], dtype=torch.float32)
    target_features = torch.tensor(target_data['features'], dtype=torch.float32)
    
    from data.preprocessing import clean_features_for_training
    clean_initial = clean_features_for_training(initial_features)
    
    # Make prediction
    with torch.no_grad():
        prediction = predictor.predict(clean_initial)
    
    # Strategy 1: Compare prediction to raw target wave features
    print(f"Strategy 1: Raw target wave features")
    if len(predictor.wave_indices) >= 3:
        true_waves_1 = target_features[:, predictor.wave_indices[:3]]
        rmse_1 = torch.sqrt(torch.mean((prediction - true_waves_1)**2))
        print(f"   RMSE: {rmse_1:.4f}m (current method)")
    
    # Strategy 2: Compare prediction to cleaned target wave features  
    print(f"Strategy 2: Cleaned target wave features")
    clean_target = clean_features_for_training(target_features)
    if len(predictor.wave_indices) >= 3:
        true_waves_2 = clean_target[:, predictor.wave_indices[:3]]
        rmse_2 = torch.sqrt(torch.mean((prediction - true_waves_2)**2))
        print(f"   RMSE: {rmse_2:.4f}m")
    
    # Strategy 3: Use training dataset targets
    print(f"Strategy 3: Training dataset targets")
    try:
        from data.datasets import SpatialWaveDataset
        
        # Find the training sample that corresponds to our time indices
        train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=10)
        
        # Get a sample that might correspond to our evaluation
        sample = train_dataset[0]  # This is t=0 -> t=1
        train_target = sample['targets']
        
        if prediction.shape == train_target.shape:
            rmse_3 = torch.sqrt(torch.mean((prediction - train_target)**2))
            print(f"   RMSE: {rmse_3:.4f}m (training dataset format)")
        else:
            print(f"   Shape mismatch: {prediction.shape} vs {train_target.shape}")
    
    except Exception as e:
        print(f"   ❌ Training dataset strategy failed: {e}")
    
    # Strategy 4: Compare to last 3 features (simple fallback)
    print(f"Strategy 4: Last 3 target features")
    true_waves_4 = target_features[:, -3:]
    rmse_4 = torch.sqrt(torch.mean((prediction - true_waves_4)**2))
    print(f"   RMSE: {rmse_4:.4f}m")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("- If Strategy 3 gives ~0.21m, use training dataset format")
    print("- If Strategy 2 is much lower, use cleaned features")
    print("- Current high RMSE suggests evaluation method mismatch")

def main():
    """Main diagnostic function"""
    
    try:
        predictor, mesh_loader = diagnose_prediction_issues()
        test_different_evaluation_strategies(predictor, mesh_loader)
        
        print(f"\n🎯 SUMMARY:")
        print("- Static predictions suggest autoregressive updates aren't working")
        print("- High RMSE suggests evaluation method mismatch")
        print("- Check which strategy gives ~0.21m RMSE")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()