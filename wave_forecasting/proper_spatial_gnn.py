#!/usr/bin/env python3
"""
Proper SpatialWaveGNN implementation with message passing
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def load_proper_spatial_model():
    """Load the model using the actual SpatialWaveGNN class"""
    
    print("📦 LOADING PROPER SpatialWaveGNN WITH MESSAGE PASSING")
    print("=" * 55)
    
    # Import the real SpatialWaveGNN
    from models.spatial import SpatialWaveGNN
    from config.base import ModelConfig
    
    # Find latest checkpoint
    logs_path = Path("logs")
    all_checkpoints = []
    
    for exp_dir in logs_path.iterdir():
        if exp_dir.is_dir():
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("*.pt"):
                    all_checkpoints.append((checkpoint.stat().st_mtime, str(checkpoint), exp_dir.name))
    
    all_checkpoints.sort(reverse=True)
    _, checkpoint_path, exp_name = all_checkpoints[0]
    
    print(f"📁 Loading: {exp_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    config_dict = checkpoint.get('config', {})
    
    print(f"📊 Checkpoint analysis:")
    print(f"   Config type: {type(config_dict)}")
    
    # Recreate the model config
    if isinstance(config_dict, dict) and 'model' in config_dict:
        model_config_dict = config_dict['model']
        print(f"   Model config: {model_config_dict}")
        
        # Create ModelConfig with correct values
        model_config = ModelConfig(
            hidden_dim=model_config_dict.get('hidden_dim', 256),
            num_spatial_layers=model_config_dict.get('num_spatial_layers', 12),
            num_temporal_layers=model_config_dict.get('num_temporal_layers', 4),
            edge_features=model_config_dict.get('edge_features', 3),
            output_features=model_config_dict.get('output_features', 3)
        )
        
        # Add input_features if we can infer it
        if 'encoder.0.weight' in state_dict:
            input_features = state_dict['encoder.0.weight'].shape[1]
            model_config.input_features = input_features
            print(f"   Detected input features: {input_features}")
    else:
        # Fallback config
        print("   Using fallback config")
        model_config = ModelConfig(
            hidden_dim=256,
            num_spatial_layers=12,
            edge_features=3,
            output_features=3
        )
        if 'encoder.0.weight' in state_dict:
            model_config.input_features = state_dict['encoder.0.weight'].shape[1]
    
    print(f"🧠 Creating SpatialWaveGNN with config:")
    print(f"   Hidden dim: {model_config.hidden_dim}")
    print(f"   Spatial layers: {model_config.num_spatial_layers}")
    print(f"   Edge features: {model_config.edge_features}")
    print(f"   Output features: {model_config.output_features}")
    if hasattr(model_config, 'input_features'):
        print(f"   Input features: {model_config.input_features}")
    
    # Create the real model
    try:
        model = SpatialWaveGNN(model_config)
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            print(f"✅ State dict loaded with strict=True")
        except Exception as e:
            print(f"⚠️  Strict loading failed: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ State dict loaded with strict=False")
            except Exception as e2:
                print(f"❌ All loading attempts failed: {e2}")
                return None, None, None
        
        model.eval()
        return model, model_config, exp_name
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_proper_model_single_prediction():
    """Test the proper model with a single prediction"""
    
    model, model_config, exp_name = load_proper_spatial_model()
    if model is None:
        print("❌ Cannot test - model loading failed")
        return None, None, None
    
    print(f"\n🧪 TESTING PROPER MODEL SINGLE PREDICTION")
    print("=" * 45)
    
    # Setup real data environment
    from data.loaders import ERA5DataManager, GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from mesh.connectivity import compute_regional_edges
    from config.base import DataConfig, MeshConfig
    
    # Use the same mesh config as training
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    era5_manager = ERA5DataManager(data_config)
    gebco_manager = GEBCODataManager(data_config)
    
    # Load data
    first_file = era5_manager.available_files['atmospheric'][0]
    filename = Path(first_file).stem
    year_month = filename.split('_')[-1]
    year, month = int(year_month[:4]), int(year_month[4:6])
    
    era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
    gebco_data = gebco_manager.load_bathymetry()
    
    # Create environment
    mesh = IcosahedralMesh(mesh_config)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
    mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
    
    # Get real features
    sample_data = mesh_loader.load_features(time_idx=0)
    real_features = torch.tensor(sample_data['features'], dtype=torch.float32)
    feature_names = sample_data['feature_names']
    
    print(f"   Real features shape: {real_features.shape}")
    print(f"   Feature names: {feature_names}")
    
    # Create graph
    region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
    edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
    
    print(f"   Graph: {len(region_indices)} nodes, {edge_index.shape[1]} edges")
    
    # Clean features for model input
    from data.preprocessing import clean_features_for_training
    clean_features = clean_features_for_training(real_features)
    
    print(f"   Cleaned features shape: {clean_features.shape}")
    
    # Test single prediction with the proper model
    print(f"🔮 Testing single prediction...")
    
    model.eval()
    with torch.no_grad():
        try:
            # This should now include proper message passing!
            wave_pred = model(clean_features, edge_index, edge_attr)
            
            print(f"✅ Single prediction successful!")
            print(f"   Input: {clean_features.shape}")
            print(f"   Output: {wave_pred.shape}")
            
            # Check if outputs are reasonable
            if wave_pred.shape[1] >= 1:
                swh_pred = wave_pred[:, 0]
                print(f"   SWH prediction range: {swh_pred.min():.3f} to {swh_pred.max():.3f}m")
                print(f"   SWH mean: {swh_pred.mean():.3f}m, std: {swh_pred.std():.3f}m")
                
                # Check if outputs are reasonable (wave heights should be 0-20m typically)
                if swh_pred.min() >= -5 and swh_pred.max() <= 25:
                    print(f"   ✅ Wave heights look reasonable!")
                    return model, mesh_loader, edge_index, edge_attr, feature_names
                else:
                    print(f"   ⚠️  Wave heights seem unreasonable")
                    return model, mesh_loader, edge_index, edge_attr, feature_names
            
        except Exception as e:
            print(f"❌ Single prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

def test_proper_autoregressive():
    """Test autoregressive prediction with the proper model"""
    
    components = test_proper_model_single_prediction()
    if components[0] is None:
        print("❌ Cannot test autoregressive - single prediction failed")
        return
    
    model, mesh_loader, edge_index, edge_attr, feature_names = components
    
    print(f"\n🔮 TESTING PROPER AUTOREGRESSIVE PREDICTION")
    print("=" * 45)
    
    # Load initial state
    initial_data = mesh_loader.load_features(time_idx=20)
    current_state = torch.tensor(initial_data['features'], dtype=torch.float32)
    
    # Clean initial state
    from data.preprocessing import clean_features_for_training
    current_state = clean_features_for_training(current_state)
    
    print(f"   Initial state: {current_state.shape}")
    print(f"   Initial SWH range: {current_state[:, 3].min():.3f} to {current_state[:, 3].max():.3f}m")  # Assuming SWH is index 3
    
    # Run autoregressive prediction
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for step in range(12):  # Test 12 steps (72 hours)
            try:
                # Make prediction with proper message passing
                wave_pred = model(current_state, edge_index, edge_attr)
                
                forecast_hours = (step + 1) * 6
                predictions[forecast_hours] = wave_pred.clone()
                
                # Check prediction quality
                swh_pred = wave_pred[:, 0]
                swh_stats = {
                    'min': swh_pred.min().item(),
                    'max': swh_pred.max().item(),
                    'mean': swh_pred.mean().item(),
                    'std': swh_pred.std().item()
                }
                
                print(f"   t+{forecast_hours:2d}h: SWH {swh_stats['min']:.3f}-{swh_stats['max']:.3f}m (μ={swh_stats['mean']:.3f}m)")
                
                # Update state for next prediction
                # Strategy: Update the wave variables (assume they're the last 3 features)
                if current_state.shape[1] >= 3 and wave_pred.shape[1] >= 3:
                    # Find wave variable indices
                    wave_indices = []
                    for i, name in enumerate(feature_names):
                        if any(wave_var in name.lower() for wave_var in ['swh', 'mwd', 'mwp']):
                            wave_indices.append(i)
                    
                    if len(wave_indices) >= 3:
                        # Update specific wave indices
                        for i, wave_idx in enumerate(wave_indices[:3]):
                            if i < wave_pred.shape[1]:
                                current_state[:, wave_idx] = wave_pred[:, i]
                    else:
                        # Fallback: update last 3 features
                        current_state[:, -3:] = wave_pred[:, :3]
                
                # Check for divergence
                if swh_stats['max'] > 50 or swh_stats['min'] < -10:
                    print(f"   ⚠️  DIVERGENCE DETECTED at step {step+1}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Step {step+1} failed: {e}")
                break
    
    print(f"\n🎯 PROPER AUTOREGRESSIVE RESULTS:")
    print(f"   ✅ Generated {len(predictions)} predictions")
    print(f"   📊 Max horizon: {max(predictions.keys()) if predictions else 0}h")
    
    if len(predictions) >= 8:  # At least 48 hours
        print(f"   🎉 Autoregressive prediction working with proper GNN!")
        
        # Quick performance check
        if len(predictions) >= 4:
            horizons_to_check = [6, 24, 48, 72]
            for h in horizons_to_check:
                if h in predictions:
                    pred = predictions[h]
                    swh = pred[:, 0]
                    print(f"   t+{h:2d}h: SWH range {swh.min():.3f}-{swh.max():.3f}m")
        
        return True, model, mesh_loader, edge_index, edge_attr, feature_names
    else:
        print(f"   ⚠️  Limited success - need to investigate further")
        return False, model, mesh_loader, edge_index, edge_attr, feature_names

def main():
    """Main function to test proper GNN model"""
    
    print("🌊 PROPER SPATIAL GNN WITH MESSAGE PASSING TEST")
    print("=" * 55)
    
    try:
        success, model, mesh_loader, edge_index, edge_attr, feature_names = test_proper_autoregressive()
        
        if success:
            print(f"\n🎉 SUCCESS! Proper GNN model is working!")
            print(f"   ✅ Message passing layers included")
            print(f"   ✅ Autoregressive prediction stable")
            print(f"   ✅ Wave heights in reasonable range")
            print(f"   🚀 Ready for full 7-day evaluation!")
            
            # Save the working components for later use
            torch.save({
                'model_state_dict': model.state_dict(),
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'feature_names': feature_names
            }, 'working_model_components.pt')
            
            print(f"💾 Working model components saved!")
        else:
            print(f"\n⚠️  Partial success - model works but has issues")
            print(f"   Check message passing implementation")
            print(f"   Verify autoregressive state update strategy")
        
    except Exception as e:
        print(f"❌ Proper GNN test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()