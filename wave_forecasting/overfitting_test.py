#!/usr/bin/env python3
"""
Test to validate overfitting hypothesis
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def test_overfitting_hypothesis():
    """Test if model performs well on training data but poorly on evaluation data"""
    
    print("🔍 TESTING OVERFITTING HYPOTHESIS")
    print("=" * 40)
    
    from prediction.forecasting import WavePredictor
    from data.datasets import SpatialWaveDataset
    from data.loaders import ERA5DataManager, GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
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
    latest_checkpoint = all_checkpoints[0][1]
    
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
    # Setup data environment
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    era5_manager = ERA5DataManager(data_config)
    gebco_manager = GEBCODataManager(data_config)
    
    # TEST 1: Performance on training-style data format
    print(f"\n📊 TEST 1: Training Data Format Performance")
    print("-" * 45)
    
    # Load 2020 data (likely training data)
    era5_atmo_2020, era5_waves_2020 = era5_manager.load_month_data(2020, 1)
    gebco_data = gebco_manager.load_bathymetry()
    
    mesh = IcosahedralMesh(mesh_config)
    interpolator_2020 = MultiResolutionInterpolator(era5_atmo_2020, era5_waves_2020, gebco_data, data_config)
    mesh_loader_2020 = MeshDataLoader(mesh, interpolator_2020, data_config)
    
    # Create training dataset
    train_dataset_2020 = SpatialWaveDataset(mesh_loader_2020, num_timesteps=10)
    
    print(f"   Testing on 2020 data (likely training): {len(train_dataset_2020)} samples")
    
    rmse_2020 = []
    for i in range(min(10, len(train_dataset_2020))):
        sample = train_dataset_2020[i]
        features = sample['features']
        targets = sample['targets']
        
        with torch.no_grad():
            prediction = predictor.predict(features)
        
        if prediction.shape == targets.shape:
            rmse = torch.sqrt(torch.mean((prediction - targets)**2))
            rmse_2020.append(rmse.item())
    
    avg_rmse_2020 = sum(rmse_2020) / len(rmse_2020) if rmse_2020 else 0
    print(f"   2020 (training) RMSE: {avg_rmse_2020:.4f}m")
    
    # TEST 2: Performance on different year data
    print(f"\n📊 TEST 2: Different Year Performance")
    print("-" * 35)
    
    try:
        # Try 2021 data (likely not in training)
        era5_atmo_2021, era5_waves_2021 = era5_manager.load_month_data(2021, 1)
        interpolator_2021 = MultiResolutionInterpolator(era5_atmo_2021, era5_waves_2021, gebco_data, data_config)
        mesh_loader_2021 = MeshDataLoader(mesh, interpolator_2021, data_config)
        
        train_dataset_2021 = SpatialWaveDataset(mesh_loader_2021, num_timesteps=10)
        
        print(f"   Testing on 2021 data (evaluation): {len(train_dataset_2021)} samples")
        
        rmse_2021 = []
        for i in range(min(10, len(train_dataset_2021))):
            sample = train_dataset_2021[i]
            features = sample['features']
            targets = sample['targets']
            
            with torch.no_grad():
                prediction = predictor.predict(features)
            
            if prediction.shape == targets.shape:
                rmse = torch.sqrt(torch.mean((prediction - targets)**2))
                rmse_2021.append(rmse.item())
        
        avg_rmse_2021 = sum(rmse_2021) / len(rmse_2021) if rmse_2021 else 0
        print(f"   2021 (evaluation) RMSE: {avg_rmse_2021:.4f}m")
        
        # Calculate overfitting ratio
        if avg_rmse_2020 > 0:
            overfitting_ratio = avg_rmse_2021 / avg_rmse_2020
            print(f"   📊 Overfitting ratio: {overfitting_ratio:.1f}x")
        
    except Exception as e:
        print(f"   ❌ 2021 data test failed: {e}")
        avg_rmse_2021 = 0
    
    # TEST 3: Spatial generalization test
    print(f"\n📊 TEST 3: Spatial Generalization")
    print("-" * 30)
    
    # Test on different spatial regions (if model was trained on specific region)
    try:
        # Create a slightly shifted spatial region
        modified_data_config = DataConfig()
        modified_data_config.lat_bounds = (15.0, 55.0)  # Shifted from (10.0, 60.0)
        modified_data_config.lon_bounds = (130.0, 230.0)  # Shifted from (120.0, 240.0)
        
        mesh_modified = IcosahedralMesh(mesh_config)
        interpolator_modified = MultiResolutionInterpolator(era5_atmo_2020, era5_waves_2020, gebco_data, modified_data_config)
        mesh_loader_modified = MeshDataLoader(mesh_modified, interpolator_modified, modified_data_config)
        
        train_dataset_modified = SpatialWaveDataset(mesh_loader_modified, num_timesteps=5)
        
        print(f"   Testing on shifted region: {len(train_dataset_modified)} samples")
        
        rmse_modified = []
        for i in range(min(5, len(train_dataset_modified))):
            sample = train_dataset_modified[i]
            features = sample['features']
            targets = sample['targets']
            
            with torch.no_grad():
                prediction = predictor.predict(features)
            
            if prediction.shape == targets.shape:
                rmse = torch.sqrt(torch.mean((prediction - targets)**2))
                rmse_modified.append(rmse.item())
        
        avg_rmse_modified = sum(rmse_modified) / len(rmse_modified) if rmse_modified else 0
        print(f"   Shifted region RMSE: {avg_rmse_modified:.4f}m")
        
    except Exception as e:
        print(f"   ❌ Spatial generalization test failed: {e}")
        avg_rmse_modified = 0
    
    # TEST 4: Feature importance analysis
    print(f"\n📊 TEST 4: Feature Sensitivity Analysis")
    print("-" * 35)
    
    try:
        # Test how sensitive the model is to different input features
        sample = train_dataset_2020[0]
        original_features = sample['features'].clone()
        
        # Test with zeroed-out features
        feature_sensitivities = []
        
        for i in range(original_features.shape[1]):
            modified_features = original_features.clone()
            modified_features[:, i] = 0  # Zero out feature i
            
            with torch.no_grad():
                original_pred = predictor.predict(original_features)
                modified_pred = predictor.predict(modified_features)
            
            # Calculate change in prediction
            pred_change = torch.mean(torch.abs(original_pred - modified_pred))
            feature_sensitivities.append((i, pred_change.item()))
        
        # Sort by sensitivity
        feature_sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Feature sensitivity (top 5):")
        for i, (feat_idx, sensitivity) in enumerate(feature_sensitivities[:5]):
            feat_name = predictor.feature_names[feat_idx] if feat_idx < len(predictor.feature_names) else f"feat_{feat_idx}"
            print(f"   {i+1}. {feat_name}: {sensitivity:.4f}")
        
        # Check if model is overly dependent on specific features
        top_sensitivity = feature_sensitivities[0][1]
        if top_sensitivity > 1.0:
            print(f"   ⚠️  High dependency on single feature - possible overfitting")
        
    except Exception as e:
        print(f"   ❌ Feature sensitivity test failed: {e}")
    
    # SUMMARY
    print(f"\n🎯 OVERFITTING ANALYSIS SUMMARY")
    print("=" * 35)
    
    print(f"Training data RMSE:     {avg_rmse_2020:.4f}m")
    print(f"Evaluation data RMSE:   {avg_rmse_2021:.4f}m")
    if avg_rmse_modified > 0:
        print(f"Spatial shift RMSE:     {avg_rmse_modified:.4f}m")
    
    # Diagnosis
    is_overfitted = False
    
    if avg_rmse_2020 < 0.5 and avg_rmse_2021 > 1.5:
        print(f"\n❌ SEVERE OVERFITTING DETECTED:")
        print(f"   - Model performs well on training data")
        print(f"   - Model fails on evaluation data")
        print(f"   - Generalization gap: {avg_rmse_2021/avg_rmse_2020 if avg_rmse_2020 > 0 else 'inf'}x")
        is_overfitted = True
    elif avg_rmse_2020 < 0.5 and avg_rmse_2021 > avg_rmse_2020 * 2:
        print(f"\n⚠️  MODERATE OVERFITTING:")
        print(f"   - Some generalization issues")
        print(f"   - Performance drops on new data")
        is_overfitted = True
    else:
        print(f"\n✅ NO SIGNIFICANT OVERFITTING:")
        print(f"   - Consistent performance across datasets")
    
    if is_overfitted:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Increase training data diversity (more years/seasons)")
        print(f"   2. Add regularization (dropout, weight decay)")
        print(f"   3. Reduce model complexity")
        print(f"   4. Use data augmentation")
        print(f"   5. Implement early stopping")
        print(f"   6. Cross-validation during training")
    
    return is_overfitted

def main():
    """Main overfitting test"""
    
    print("🧪 OVERFITTING VALIDATION TEST")
    print("=" * 50)
    
    try:
        is_overfitted = test_overfitting_hypothesis()
        
        if is_overfitted:
            print(f"\n🎯 CONCLUSION: Model is overfitted to training data")
            print(f"   Autoregressive performance issues are due to poor generalization")
            print(f"   Need to retrain with better regularization and more diverse data")
        else:
            print(f"\n🎯 CONCLUSION: Overfitting is not the primary issue")
            print(f"   Look for other causes (data preprocessing, evaluation method, etc.)")
        
    except Exception as e:
        print(f"❌ Overfitting test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()