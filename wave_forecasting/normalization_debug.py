#!/usr/bin/env python3
"""
Debug normalization issues in the robust model
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from test_retrain import FeatureNormalizer
from test_retrain import (
    RobustSpatialWaveGNN, 
    RobustTrainingConfig, 
    FeatureNormalizer,
    ImprovedWaveLoss
)


def debug_normalization():
    """Debug the normalization process"""
    
    print("🔍 DEBUGGING NORMALIZATION ISSUES")
    print("=" * 50)
    
    # Load model checkpoint
    model_path = "logs/robust_spatial_20250702_215103/final_robust_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract normalizer
    normalizer_state = checkpoint['normalizer_state']
    normalizer = FeatureNormalizer()
    normalizer.feature_scaler = normalizer_state['feature_scaler']
    normalizer.target_scaler = normalizer_state['target_scaler']
    normalizer.fitted = True
    
    print("📊 Normalizer Statistics:")
    print(f"   Feature scaler type: {type(normalizer.feature_scaler)}")
    print(f"   Target scaler type: {type(normalizer.target_scaler)}")
    
    # Check feature scaler
    if hasattr(normalizer.feature_scaler, 'center_'):
        print(f"   Feature center (first 5): {normalizer.feature_scaler.center_[:5]}")
        print(f"   Feature scale (first 5): {normalizer.feature_scaler.scale_[:5]}")
    
    # Check target scaler
    if hasattr(normalizer.target_scaler, 'mean_'):
        print(f"   Target mean: {normalizer.target_scaler.mean_}")
        print(f"   Target scale: {normalizer.target_scaler.scale_}")
    
    # Test the normalization process
    print(f"\n🧪 Testing Normalization Process:")
    
    # Create sample realistic targets
    realistic_targets = np.array([
        [2.5, 180.0, 8.5],   # Reasonable wave: 2.5m SWH, 180° MWD, 8.5s MWP
        [1.0, 90.0, 6.0],    # Small wave
        [5.0, 270.0, 12.0],  # Larger wave
        [0.5, 45.0, 4.0]     # Very small wave
    ])
    
    print(f"   Original targets:")
    for i, target in enumerate(realistic_targets):
        print(f"     Sample {i+1}: SWH={target[0]:.1f}m, MWD={target[1]:.1f}°, MWP={target[2]:.1f}s")
    
    # Normalize targets
    normalized_targets = normalizer.transform_targets(realistic_targets)
    print(f"\n   Normalized targets:")
    for i, target in enumerate(normalized_targets):
        print(f"     Sample {i+1}: [{target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}]")
    
    # Denormalize back
    denormalized_targets = normalizer.inverse_transform_targets(normalized_targets)
    print(f"\n   Denormalized targets:")
    for i, target in enumerate(denormalized_targets):
        print(f"     Sample {i+1}: SWH={target[0]:.1f}m, MWD={target[1]:.1f}°, MWP={target[2]:.1f}s")
    
    # Check if round-trip works
    round_trip_error = np.mean(np.abs(realistic_targets - denormalized_targets))
    print(f"\n   Round-trip error: {round_trip_error:.8f}")
    
    if round_trip_error > 1e-6:
        print(f"   ❌ PROBLEM: Normalization round-trip failed!")
    else:
        print(f"   ✅ Normalization round-trip works")
    
    # Test what happens with model-scale predictions
    print(f"\n🔬 Testing Model-Scale Predictions:")
    
    # Simulate what the model might be outputting (in normalized space)
    model_outputs = np.array([
        [0.5, -0.2, 1.1],    # Some normalized predictions
        [-1.0, 0.8, -0.5],
        [2.0, -1.5, 0.0]
    ])
    
    print(f"   Model outputs (normalized space):")
    for i, output in enumerate(model_outputs):
        print(f"     Output {i+1}: [{output[0]:.6f}, {output[1]:.6f}, {output[2]:.6f}]")
    
    # Denormalize model outputs
    denorm_outputs = normalizer.inverse_transform_targets(model_outputs)
    print(f"\n   Denormalized outputs:")
    for i, output in enumerate(denorm_outputs):
        print(f"     Output {i+1}: SWH={output[0]:.1f}m, MWD={output[1]:.1f}°, MWP={output[2]:.1f}s")
    
    return normalizer

def test_fixed_prediction():
    """Test prediction with fixed normalization"""
    
    print(f"\n🔧 TESTING FIXED PREDICTION")
    print("=" * 40)
    
    normalizer = debug_normalization()
    
    # Load the robust model
    from robust_model_eval import RobustWavePredictor
    
    model_path = "logs/robust_spatial_20250702_215103/final_robust_model.pt"
    
    class FixedRobustPredictor(RobustWavePredictor):
        """Fixed version with proper normalization handling"""
        
        def predict_fixed(self, features: torch.Tensor) -> torch.Tensor:
            """Fixed prediction method"""
            
            features = features.to(self.device)
            
            # Normalize features
            features_np = features.cpu().numpy()
            features_normalized = self.normalizer.transform_features(features_np)
            features = torch.tensor(features_normalized, dtype=torch.float32, device=self.device)
            
            # Make prediction (model outputs in normalized space)
            with torch.no_grad():
                predictions = self.model(features, self.edge_index, self.edge_attr)
            
            # Check prediction range before denormalization
            print(f"     Model raw output range: {predictions.min():.6f} to {predictions.max():.6f}")
            
            # Denormalize predictions
            predictions_np = predictions.cpu().numpy()
            predictions_denorm = self.normalizer.inverse_transform_targets(predictions_np)
            
            return torch.tensor(predictions_denorm, dtype=torch.float32)
    
    try:
        # Create fixed predictor
        predictor = FixedRobustPredictor(model_path)
        
        # Load real data
        real_data = predictor.load_real_data_sample(time_idx=10)
        features = real_data['features']
        
        print(f"   Real data feature range: {features.min():.3f} to {features.max():.3f}")
        
        # Make fixed prediction
        pred_fixed = predictor.predict_fixed(features)
        
        # Check results
        swh_values = pred_fixed[:, 0]
        mwd_values = pred_fixed[:, 1]
        mwp_values = pred_fixed[:, 2]
        
        print(f"\n   📊 Fixed Prediction Results:")
        print(f"   SWH: {swh_values.mean():.3f} ± {swh_values.std():.3f} m (range: {swh_values.min():.3f} to {swh_values.max():.3f})")
        print(f"   MWD: {mwd_values.mean():.1f} ± {mwd_values.std():.1f} deg (range: {mwd_values.min():.1f} to {mwd_values.max():.1f})")
        print(f"   MWP: {mwp_values.mean():.3f} ± {mwp_values.std():.3f} s (range: {mwp_values.min():.3f} to {mwp_values.max():.3f})")
        
        # Check physical reasonableness
        reasonable_swh = torch.all(swh_values >= -5) and torch.all(swh_values <= 25)  # Allow some slack
        reasonable_mwd = torch.all(mwd_values >= -50) and torch.all(mwd_values <= 400)  # Allow some slack
        reasonable_mwp = torch.all(mwp_values >= -2) and torch.all(mwp_values <= 30)   # Allow some slack
        
        if reasonable_swh and reasonable_mwd and reasonable_mwp:
            print(f"   ✅ Fixed predictions are more reasonable!")
        else:
            print(f"   ❌ Still having issues:")
            print(f"      SWH reasonable: {reasonable_swh}")
            print(f"      MWD reasonable: {reasonable_mwd}")
            print(f"      MWP reasonable: {reasonable_mwp}")
            
    except Exception as e:
        print(f"   ❌ Fixed prediction test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function"""
    
    debug_normalization()
    test_fixed_prediction()

if __name__ == "__main__":
    main()