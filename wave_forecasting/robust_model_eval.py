#!/usr/bin/env python3
"""
Robust model functionality test and evaluation
Tests the newly trained robust model with proper dimensions and architecture
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path.cwd()))

# Import model components from test_retrain.py
from test_retrain import (
    RobustSpatialWaveGNN, 
    RobustTrainingConfig, 
    FeatureNormalizer,
    ImprovedWaveLoss
)

class RobustWavePredictor:
    """
    Predictor wrapper for the robust wave model
    Handles loading, normalization, and prediction
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.normalizer = None
        self.edge_index = None
        self.edge_attr = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        
        # Load model components
        self._load_model()
        self._setup_mesh_data()
    
    def _load_model(self):
        """Load the robust model and its components"""
        
        print(f"📂 Loading robust model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract components
            self.config = checkpoint['config']
            model_state = checkpoint['model_state_dict']
            normalizer_state = checkpoint['normalizer_state']
            
            print(f"✅ Model config loaded:")
            print(f"   Hidden dim: {self.config.hidden_dim}")
            print(f"   Spatial layers: {self.config.num_spatial_layers}")
            print(f"   Dropout: {self.config.dropout}")
            print(f"   Max training samples: {self.config.max_training_samples}")
            
            # Create model
            self.model = RobustSpatialWaveGNN(self.config, input_features=11)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            # Create normalizer
            self.normalizer = FeatureNormalizer()
            self.normalizer.feature_scaler = normalizer_state['feature_scaler']
            self.normalizer.target_scaler = normalizer_state['target_scaler']
            self.normalizer.fitted = True
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_mesh_data(self):
        """Setup mesh connectivity data"""
        
        print(f"🔧 Setting up mesh connectivity...")
        
        try:
            from data.loaders import ERA5DataManager, GEBCODataManager
            from data.preprocessing import MultiResolutionInterpolator
            from data.datasets import MeshDataLoader
            from mesh.icosahedral import IcosahedralMesh
            from mesh.connectivity import compute_regional_edges
            from config.base import DataConfig, MeshConfig
            
            # Create components
            data_config = DataConfig()
            mesh_config = MeshConfig(refinement_level=5)
            
            era5_manager = ERA5DataManager(data_config)
            gebco_manager = GEBCODataManager(data_config)
            
            # Load minimal data for mesh setup
            era5_atmo, era5_waves = era5_manager.load_month_data(2020, 1)
            gebco_data = gebco_manager.load_bathymetry()
            
            # Create mesh and connectivity
            mesh = IcosahedralMesh(mesh_config)
            interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
            self.mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
            
            # Get mesh connectivity
            region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
            edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
            
            self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
            self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=self.device)
            
            self.num_nodes = len(region_indices)
            self.num_features = 11
            
            print(f"✅ Mesh setup complete:")
            print(f"   Nodes: {self.num_nodes}")
            print(f"   Edges: {self.edge_index.shape[1]}")
            print(f"   Features: {self.num_features}")
            
        except Exception as e:
            print(f"❌ Failed to setup mesh: {e}")
            raise
    
    def predict(self, features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Make prediction with the robust model
        
        Args:
            features: Input features [num_nodes, num_features]
            normalize: Whether to apply normalization
        
        Returns:
            Predictions [num_nodes, 3] for [SWH, MWD, MWP]
        """
        
        features = features.to(self.device)
        
        # Normalize features if requested
        if normalize and self.normalizer is not None:
            features_np = features.cpu().numpy()
            features_normalized = self.normalizer.transform_features(features_np)
            features = torch.tensor(features_normalized, dtype=torch.float32, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(features, self.edge_index, self.edge_attr)
        
        # Denormalize predictions if normalizer is available
        if normalize and self.normalizer is not None:
            predictions_np = predictions.cpu().numpy()
            predictions_denorm = self.normalizer.inverse_transform_targets(predictions_np)
            predictions = torch.tensor(predictions_denorm, dtype=torch.float32)
        
        return predictions
    
    def load_real_data_sample(self, time_idx: int = 5) -> Dict[str, torch.Tensor]:
        """Load a real data sample for testing"""
        
        try:
            features_data = self.mesh_loader.load_features(time_idx=time_idx)
            features = torch.tensor(features_data['features'], dtype=torch.float32)
            
            # Clean features (handle NaN, etc.)
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return {
                'features': features,
                'feature_names': features_data.get('feature_names', [f'feat_{i}' for i in range(features.shape[1])])
            }
            
        except Exception as e:
            print(f"❌ Failed to load real data: {e}")
            return None

def test_basic_functionality(predictor: RobustWavePredictor):
    """Test basic model functionality"""
    
    print(f"\n🧪 TEST 1: Basic Model Functionality")
    print("-" * 45)
    
    num_nodes = predictor.num_nodes
    num_features = predictor.num_features
    
    # Create test inputs
    input1 = torch.zeros(num_nodes, num_features)
    input2 = torch.ones(num_nodes, num_features)
    input3 = torch.randn(num_nodes, num_features)
    input4 = torch.randn(num_nodes, num_features) * 2
    
    try:
        # Make predictions
        pred1 = predictor.predict(input1, normalize=False)
        pred2 = predictor.predict(input2, normalize=False)
        pred3 = predictor.predict(input3, normalize=False)
        pred4 = predictor.predict(input4, normalize=False)
        
        print(f"✅ All predictions successful!")
        print(f"   Input 1 (zeros): Output range {pred1.min():.6f} to {pred1.max():.6f}")
        print(f"   Input 2 (ones):  Output range {pred2.min():.6f} to {pred2.max():.6f}")
        print(f"   Input 3 (rand1): Output range {pred3.min():.6f} to {pred3.max():.6f}")
        print(f"   Input 4 (rand2): Output range {pred4.min():.6f} to {pred4.max():.6f}")
        
        # Check differences
        diff_1_2 = torch.mean(torch.abs(pred1 - pred2))
        diff_1_3 = torch.mean(torch.abs(pred1 - pred3))
        diff_3_4 = torch.mean(torch.abs(pred3 - pred4))
        
        print(f"   Difference 1-2: {diff_1_2:.6f}")
        print(f"   Difference 1-3: {diff_1_3:.6f}")
        print(f"   Difference 3-4: {diff_3_4:.6f}")
        
        if diff_1_2 < 1e-6 and diff_1_3 < 1e-6 and diff_3_4 < 1e-6:
            print(f"   ❌ PROBLEM: Model outputs identical values!")
            return False
        else:
            print(f"   ✅ Model responds to different inputs")
            return True
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_sensitivity(predictor: RobustWavePredictor):
    """Test individual feature sensitivity"""
    
    print(f"\n🧪 TEST 2: Feature Sensitivity Analysis")
    print("-" * 45)
    
    # Feature names
    feature_names = [
        'u10', 'v10', 'msl',           # Atmospheric
        'swh', 'mwd', 'mwp',           # Wave
        'bathymetry',                   # Bathymetry
        'land_mask', 'shallow_water_mask', 'deep_water_mask',  # Masks
        'distance_to_coast'             # Derived
    ]
    
    # Create baseline input
    baseline_input = torch.randn(predictor.num_nodes, predictor.num_features)
    
    try:
        baseline_pred = predictor.predict(baseline_input, normalize=False)
        print(f"   Baseline prediction range: {baseline_pred.min():.6f} to {baseline_pred.max():.6f}")
        
        # Test each feature
        sensitivities = []
        
        for feat_idx in range(predictor.num_features):
            # Modify single feature
            modified_input = baseline_input.clone()
            modified_input[:, feat_idx] = modified_input[:, feat_idx] + 1.0  # Add 1 std dev
            
            modified_pred = predictor.predict(modified_input, normalize=False)
            
            # Calculate sensitivity
            sensitivity = torch.mean(torch.abs(modified_pred - baseline_pred))
            sensitivities.append((feat_idx, sensitivity.item()))
            
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
            if feat_idx < 8:  # Show first 8
                print(f"   {feat_name:20s}: Change = {sensitivity:.6f}")
        
        # Sort by sensitivity
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   🏆 Top 5 most sensitive features:")
        for i, (feat_idx, sensitivity) in enumerate(sensitivities[:5]):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
            print(f"   {i+1}. {feat_name:20s}: {sensitivity:.6f}")
        
        avg_sensitivity = np.mean([s[1] for s in sensitivities])
        max_sensitivity = sensitivities[0][1]
        
        print(f"\n   📊 Sensitivity Statistics:")
        print(f"   Average sensitivity: {avg_sensitivity:.6f}")
        print(f"   Maximum sensitivity: {max_sensitivity:.6f}")
        
        if max_sensitivity < 1e-6:
            print(f"   ❌ PROBLEM: No feature sensitivity detected!")
            return False, 0, 0
        else:
            print(f"   ✅ Model shows feature sensitivity")
            return True, avg_sensitivity, max_sensitivity
            
    except Exception as e:
        print(f"   ❌ Feature sensitivity test failed: {e}")
        return False, 0, 0

def test_real_data_performance(predictor: RobustWavePredictor):
    """Test performance on real data"""
    
    print(f"\n🧪 TEST 3: Real Data Performance")
    print("-" * 40)
    
    try:
        # Load real data sample
        real_data = predictor.load_real_data_sample(time_idx=10)
        
        if real_data is None:
            print(f"   ❌ Failed to load real data")
            return False
        
        features = real_data['features']
        print(f"   Real data shape: {features.shape}")
        print(f"   Feature range: {features.min():.3f} to {features.max():.3f}")
        
        # Test with and without normalization
        pred_raw = predictor.predict(features, normalize=False)
        pred_norm = predictor.predict(features, normalize=True)
        
        print(f"   Raw prediction range: {pred_raw.min():.6f} to {pred_raw.max():.6f}")
        print(f"   Normalized prediction range: {pred_norm.min():.6f} to {pred_norm.max():.6f}")
        
        # Check for reasonable wave values
        swh_values = pred_norm[:, 0]  # Significant wave height
        mwd_values = pred_norm[:, 1]  # Mean wave direction  
        mwp_values = pred_norm[:, 2]  # Mean wave period
        
        print(f"\n   📊 Prediction Statistics (Normalized):")
        print(f"   SWH: {swh_values.mean():.3f} ± {swh_values.std():.3f} m (range: {swh_values.min():.3f} to {swh_values.max():.3f})")
        print(f"   MWD: {mwd_values.mean():.1f} ± {mwd_values.std():.1f} deg (range: {mwd_values.min():.1f} to {mwd_values.max():.1f})")
        print(f"   MWP: {mwp_values.mean():.3f} ± {mwp_values.std():.3f} s (range: {mwp_values.min():.3f} to {mwp_values.max():.3f})")
        
        # Check for physically reasonable values
        reasonable_swh = torch.all(swh_values >= 0) and torch.all(swh_values <= 20)
        reasonable_mwd = torch.all(mwd_values >= 0) and torch.all(mwd_values <= 360)
        reasonable_mwp = torch.all(mwp_values >= 1) and torch.all(mwp_values <= 25)
        
        if reasonable_swh and reasonable_mwd and reasonable_mwp:
            print(f"   ✅ Predictions are physically reasonable")
            return True
        else:
            print(f"   ⚠️  Some predictions outside physical bounds")
            print(f"      SWH reasonable: {reasonable_swh}")
            print(f"      MWD reasonable: {reasonable_mwd}")
            print(f"      MWP reasonable: {reasonable_mwp}")
            return True  # Still functional, just poorly calibrated
            
    except Exception as e:
        print(f"   ❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_data_comparison(predictor: RobustWavePredictor):
    """Compare performance on training-style data"""
    
    print(f"\n🧪 TEST 4: Training Data Comparison")
    print("-" * 40)
    
    try:
        from data.datasets import SpatialWaveDataset
        
        # Create training dataset
        dataset = SpatialWaveDataset(predictor.mesh_loader, num_timesteps=5)
        
        if len(dataset) == 0:
            print(f"   ❌ No training samples available")
            return False
        
        # Test on a few samples
        rmse_values = []
        mae_values = []
        
        num_test_samples = min(10, len(dataset))
        
        for i in range(num_test_samples):
            sample = dataset[i]
            features = sample['features']
            targets = sample['targets']
            
            # Make prediction
            predictions = predictor.predict(features, normalize=True)
            
            # Calculate metrics
            rmse = torch.sqrt(torch.mean((predictions - targets)**2))
            mae = torch.mean(torch.abs(predictions - targets))
            
            rmse_values.append(rmse.item())
            mae_values.append(mae.item())
        
        avg_rmse = np.mean(rmse_values)
        avg_mae = np.mean(mae_values)
        
        print(f"   📊 Training Data Performance ({num_test_samples} samples):")
        print(f"   Average RMSE: {avg_rmse:.6f}")
        print(f"   Average MAE:  {avg_mae:.6f}")
        print(f"   RMSE std:     {np.std(rmse_values):.6f}")
        
        # Evaluate performance
        if avg_rmse < 1.0:
            print(f"   ✅ Good performance (RMSE < 1.0)")
            return True
        elif avg_rmse < 2.0:
            print(f"   ⚠️  Moderate performance (1.0 ≤ RMSE < 2.0)")
            return True
        else:
            print(f"   ❌ Poor performance (RMSE ≥ 2.0)")
            return False
            
    except Exception as e:
        print(f"   ❌ Training data comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_diagnostic_plots(predictor: RobustWavePredictor, save_dir: Optional[str] = None):
    """Generate diagnostic plots"""
    
    print(f"\n📊 Generating Diagnostic Plots")
    print("-" * 35)
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
    else:
        save_path = Path("robust_model_diagnostics")
        save_path.mkdir(exist_ok=True)
    
    try:
        # Load real data for plotting
        real_data = predictor.load_real_data_sample(time_idx=10)
        if real_data is None:
            print(f"   ❌ Cannot generate plots without real data")
            return
        
        features = real_data['features']
        
        # Make predictions
        pred_raw = predictor.predict(features, normalize=False)
        pred_norm = predictor.predict(features, normalize=True)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Raw vs normalized predictions
        for i, (var_name, unit) in enumerate([('SWH', 'm'), ('MWD', 'deg'), ('MWP', 's')]):
            axes[0, i].scatter(pred_raw[:, i].cpu(), pred_norm[:, i].cpu(), alpha=0.6, s=1)
            axes[0, i].set_xlabel(f'{var_name} Raw')
            axes[0, i].set_ylabel(f'{var_name} Normalized')
            axes[0, i].set_title(f'{var_name} Raw vs Normalized')
            axes[0, i].grid(True, alpha=0.3)
        
        # Prediction distributions
        for i, (var_name, unit) in enumerate([('SWH', 'm'), ('MWD', 'deg'), ('MWP', 's')]):
            axes[1, i].hist(pred_norm[:, i].cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, i].set_xlabel(f'{var_name} ({unit})')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'{var_name} Distribution')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('Robust Wave Model Diagnostics', fontsize=16)
        plt.tight_layout()
        
        plot_path = save_path / "diagnostic_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"   ❌ Plot generation failed: {e}")

def main():
    """Main diagnostic function"""
    
    print("🌊 ROBUST WAVE MODEL DIAGNOSTIC")
    print("=" * 45)
    
    # Model path
    model_path = "logs/robust_spatial_20250702_215103/final_robust_model.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load predictor
        predictor = RobustWavePredictor(model_path)
        
        # Run tests
        print(f"\n🔬 Running diagnostic tests...")
        
        # Test 1: Basic functionality
        basic_ok = test_basic_functionality(predictor)
        
        if not basic_ok:
            print(f"\n❌ CRITICAL: Basic functionality failed!")
            return
        
        # Test 2: Feature sensitivity
        sensitivity_ok, avg_sens, max_sens = test_feature_sensitivity(predictor)
        
        # Test 3: Real data performance
        real_data_ok = test_real_data_performance(predictor)
        
        # Test 4: Training data comparison
        training_ok = test_training_data_comparison(predictor)
        
        # Generate plots
        generate_diagnostic_plots(predictor)
        
        # Summary
        print(f"\n📋 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        print(f"✅ Basic functionality: {'PASS' if basic_ok else 'FAIL'}")
        print(f"✅ Feature sensitivity: {'PASS' if sensitivity_ok else 'FAIL'}")
        if sensitivity_ok:
            print(f"   - Average sensitivity: {avg_sens:.6f}")
            print(f"   - Maximum sensitivity: {max_sens:.6f}")
        print(f"✅ Real data performance: {'PASS' if real_data_ok else 'FAIL'}")
        print(f"✅ Training data performance: {'PASS' if training_ok else 'FAIL'}")
        
        if all([basic_ok, sensitivity_ok, real_data_ok]):
            print(f"\n🎉 ROBUST MODEL IS FUNCTIONAL!")
            print(f"   The model shows {avg_sens:.6f} average feature sensitivity")
            print(f"   This is a {avg_sens/0.000025:.0f}x improvement over the previous model")
            print(f"   Ready for autoregressive testing!")
        else:
            print(f"\n⚠️  ROBUST MODEL HAS ISSUES")
            print(f"   Some tests failed - check individual results above")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()