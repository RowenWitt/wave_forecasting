#!/usr/bin/env python3
"""
Fixed model functionality test with correct node dimensions
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def test_model_with_correct_dimensions():
    """Test model functionality with the correct node count"""
    
    print("🔍 TESTING MODEL WITH CORRECT DIMENSIONS")
    print("=" * 45)
    
    from prediction.forecasting import WavePredictor
    
    # Load model
    logs_path = Path("logs")
    all_checkpoints = []
    for exp_dir in logs_path.iterdir():
        if exp_dir.is_dir():
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("*.pt"):
                    all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint)))
    
    all_checkpoints.sort(reverse=True)
    print(all_checkpoints)
    latest_checkpoint = all_checkpoints[0][1]
    
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
    # Get correct dimensions from the model
    edge_index = predictor.edge_index
    max_node_idx = edge_index.max().item()
    min_node_idx = edge_index.min().item()
    num_nodes = max_node_idx + 1  # Correct number of nodes
    num_features = len(predictor.feature_names)
    
    print(f"📊 Model dimensions:")
    print(f"   Nodes: {num_nodes}")
    print(f"   Features: {num_features}")
    print(f"   Edges: {edge_index.shape[1]}")
    print(f"   Node index range: {min_node_idx} to {max_node_idx}")
    
    # TEST 1: Input sensitivity with correct dimensions
    print(f"\n🧪 TEST 1: Input Sensitivity (Correct Dimensions)")
    print("-" * 50)
    
    # Create test inputs with correct dimensions
    input1 = torch.zeros(num_nodes, num_features)
    input2 = torch.ones(num_nodes, num_features)
    input3 = torch.randn(num_nodes, num_features)
    input4 = torch.randn(num_nodes, num_features) * 5
    
    # Make predictions
    try:
        with torch.no_grad():
            pred1 = predictor.predict(input1)
            pred2 = predictor.predict(input2)
            pred3 = predictor.predict(input3)
            pred4 = predictor.predict(input4)
        
        print(f"   ✅ All predictions successful!")
        print(f"   Input 1 (zeros): Output range {pred1.min():.6f} to {pred1.max():.6f}")
        print(f"   Input 2 (ones):  Output range {pred2.min():.6f} to {pred2.max():.6f}")
        print(f"   Input 3 (rand1): Output range {pred3.min():.6f} to {pred3.max():.6f}")
        print(f"   Input 4 (rand2): Output range {pred4.min():.6f} to {pred4.max():.6f}")
        
        # Check if outputs are different
        diff_1_2 = torch.mean(torch.abs(pred1 - pred2))
        diff_1_3 = torch.mean(torch.abs(pred1 - pred3))
        diff_3_4 = torch.mean(torch.abs(pred3 - pred4))
        
        print(f"   Difference 1-2: {diff_1_2:.6f}")
        print(f"   Difference 1-3: {diff_1_3:.6f}")
        print(f"   Difference 3-4: {diff_3_4:.6f}")
        
        if diff_1_2 < 1e-6 and diff_1_3 < 1e-6 and diff_3_4 < 1e-6:
            print(f"   ❌ PROBLEM: Model outputs identical values for different inputs!")
            return False, num_nodes, num_features
        else:
            print(f"   ✅ Model responds to different inputs")
            return True, num_nodes, num_features
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False, num_nodes, num_features

def test_individual_feature_changes(predictor, num_nodes, num_features):
    """Test how the model responds to changes in individual features"""
    
    print(f"\n🧪 TEST 2: Individual Feature Sensitivity")
    print("-" * 40)
    
    # Create a baseline input
    baseline_input = torch.randn(num_nodes, num_features)
    
    try:
        with torch.no_grad():
            baseline_pred = predictor.predict(baseline_input)
        
        print(f"   Baseline prediction range: {baseline_pred.min():.6f} to {baseline_pred.max():.6f}")
        
        # Test changing each feature individually
        feature_sensitivities = []
        
        for feat_idx in range(num_features):
            # Create modified input (change only one feature)
            modified_input = baseline_input.clone()
            modified_input[:, feat_idx] = modified_input[:, feat_idx] + 10.0  # Large change
            
            with torch.no_grad():
                modified_pred = predictor.predict(modified_input)
            
            # Calculate change in prediction
            pred_change = torch.mean(torch.abs(modified_pred - baseline_pred))
            feature_sensitivities.append((feat_idx, pred_change.item()))
            
            if feat_idx < 5:  # Show first 5 for brevity
                feat_name = predictor.feature_names[feat_idx] if feat_idx < len(predictor.feature_names) else f"feat_{feat_idx}"
                print(f"   {feat_name:15s}: Change = {pred_change:.6f}")
        
        # Sort by sensitivity
        feature_sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   Top 5 most sensitive features:")
        for i, (feat_idx, sensitivity) in enumerate(feature_sensitivities[:5]):
            feat_name = predictor.feature_names[feat_idx] if feat_idx < len(predictor.feature_names) else f"feat_{feat_idx}"
            print(f"   {i+1}. {feat_name:15s}: {sensitivity:.6f}")
        
        # Check if any features cause changes
        max_sensitivity = feature_sensitivities[0][1]
        if max_sensitivity < 1e-6:
            print(f"   ❌ PROBLEM: No features cause output changes!")
            return False
        else:
            print(f"   ✅ Model is sensitive to feature changes")
            return True
            
    except Exception as e:
        print(f"   ❌ Feature sensitivity test failed: {e}")
        return False

def test_real_vs_training_data(predictor):
    """Compare model performance on real data vs training-style data"""
    
    print(f"\n🧪 TEST 3: Real Data vs Training Data Performance")
    print("-" * 50)
    
    try:
        # Setup real data environment
        from data.loaders import ERA5DataManager, GEBCODataManager
        from data.preprocessing import MultiResolutionInterpolator
        from data.datasets import MeshDataLoader, SpatialWaveDataset
        from mesh.icosahedral import IcosahedralMesh
        from config.base import DataConfig, MeshConfig
        
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load 2020 data
        era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
        gebco_data = gebco_manager.load_bathymetry()
        
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Test 1: Direct mesh loader features
        print(f"   Testing direct mesh loader features...")
        features_data = mesh_loader.load_features(time_idx=5)  # Use valid time
        real_features = torch.tensor(features_data['features'], dtype=torch.float32)
        
        from data.preprocessing import clean_features_for_training
        clean_real_features = clean_features_for_training(real_features)
        
        with torch.no_grad():
            real_pred = predictor.predict(clean_real_features)
        
        print(f"     Real data prediction: {real_pred.min():.6f} to {real_pred.max():.6f}")
        print(f"     Real data shape: {clean_real_features.shape}")
        
        # Test 2: Training dataset format
        print(f"   Testing training dataset format...")
        train_dataset = SpatialWaveDataset(mesh_loader, num_timesteps=5)
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            train_features = sample['features']
            train_targets = sample['targets']
            
            with torch.no_grad():
                train_pred = predictor.predict(train_features)
            
            # Calculate RMSE
            rmse = torch.sqrt(torch.mean((train_pred - train_targets)**2))
            
            print(f"     Training format prediction: {train_pred.min():.6f} to {train_pred.max():.6f}")
            print(f"     Training format RMSE: {rmse:.6f}")
            print(f"     Training targets: {train_targets.min():.6f} to {train_targets.max():.6f}")
            
            # Check if real data and training data give similar predictions
            real_vs_train_diff = torch.mean(torch.abs(real_pred - train_pred))
            print(f"     Real vs Training prediction diff: {real_vs_train_diff:.6f}")
            
            if rmse > 5.0:
                print(f"   ❌ PROBLEM: Very high RMSE suggests model issues")
                return False
            elif rmse > 1.0:
                print(f"   ⚠️  High RMSE: Model may not be well-trained")
                return True
            else:
                print(f"   ✅ Reasonable RMSE: Model appears functional")
                return True
        else:
            print(f"   ❌ No training dataset samples available")
            return False
            
    except Exception as e:
        print(f"   ❌ Real vs training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main fixed diagnostic"""
    
    print("🔧 FIXED MODEL FUNCTIONALITY DIAGNOSTIC")
    print("=" * 50)
    
    try:
        # Test with correct dimensions
        input_sensitivity_ok, num_nodes, num_features = test_model_with_correct_dimensions()
        
        if not input_sensitivity_ok:
            print(f"\n❌ INPUT SENSITIVITY FAILED")
            print(f"   Model is not responding to different inputs")
            print(f"   This suggests the model is broken or weights didn't load")
            return
        
        # Get predictor for further tests
        logs_path = Path("logs")
        all_checkpoints = []
        for exp_dir in logs_path.iterdir():
            if exp_dir.is_dir():
                checkpoint_dir = exp_dir / "checkpoints"
                if checkpoint_dir.exists():
                    for checkpoint in checkpoint_dir.glob("*.pt"):
                        all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint)))
        
        all_checkpoints.sort(reverse=True)
        latest_checkpoint = all_checkpoints[-1][1]
        
        from prediction.forecasting import WavePredictor
        predictor = WavePredictor.from_checkpoint(latest_checkpoint)
        
        # Test feature sensitivity
        feature_sensitivity_ok = test_individual_feature_changes(predictor, num_nodes, num_features)
        
        if not feature_sensitivity_ok:
            print(f"\n❌ FEATURE SENSITIVITY FAILED")
            print(f"   Model doesn't respond to feature changes")
            return
        
        # Test real data performance
        real_data_ok = test_real_vs_training_data(predictor)
        
        if real_data_ok:
            print(f"\n✅ MODEL FUNCTIONALITY CONFIRMED")
            print(f"   Model is working and responding to inputs")
            print(f"   The ~1.0m RMSE issue is likely due to:")
            print(f"   - Model genuinely has ~1.0m performance (not 0.21m)")
            print(f"   - Training reported performance was optimistic")
            print(f"   - Different evaluation methodology")
            print(f"   - Model needs more training or better architecture")
        else:
            print(f"\n⚠️  MODEL ISSUES DETECTED")
            print(f"   Model responds to inputs but performance is poor")
            print(f"   Consider retraining with better architecture/data")
        
    except Exception as e:
        print(f"❌ Fixed diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()