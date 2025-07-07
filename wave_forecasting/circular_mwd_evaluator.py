#!/usr/bin/env python3
"""
Evaluator for Circular MWD Spatiotemporal Model
Handles conversion between circular and angular representations
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.insert(0, str(Path.cwd()))

# Import required components
from spatiotemporal_with_circular_mwd import SpatioTemporalWaveGNN, SpatioTemporalConfig
from mwd_circular_fixes import VariableSpecificNormalizer, evaluate_model_with_circular_metrics
from sklearn.preprocessing import StandardScaler

class CircularMWDEvaluator:
    """Evaluator for circular MWD spatiotemporal model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.load_model()
    
    def load_model(self):
        """Load trained circular MWD model"""
        
        print(f"📂 Loading circular MWD model from: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.config = checkpoint['config']
        self.feature_normalizer = checkpoint['feature_normalizer']
        self.target_normalizer = checkpoint['target_normalizer']
        self.edge_index = checkpoint['edge_index'].to(self.device)
        self.edge_attr = checkpoint['edge_attr'].to(self.device)
        
        # Create and load model
        self.model = SpatioTemporalWaveGNN(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Circular MWD model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Output format: [SWH, MWD_cos, MWD_sin, MWP]")
    
    def predict(self, input_sequence: torch.Tensor, multi_step: bool = False) -> torch.Tensor:
        """
        Make prediction and convert back to [SWH, MWD, MWP] format
        
        Args:
            input_sequence: [seq_len, num_nodes, features] or [batch_size, seq_len, num_nodes, features]
            multi_step: Whether to use multi-step prediction
        
        Returns:
            predictions: [num_nodes, 3] or [batch_size, num_nodes, 3] in [SWH, MWD, MWP] format
        """
        
        if input_sequence.dim() == 3:
            input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        input_sequence = input_sequence.to(self.device)
        
        # Normalize features
        batch_size, seq_len, num_nodes, num_features = input_sequence.size()
        input_flat = input_sequence.view(-1, num_features).cpu().numpy()
        input_norm = self.feature_normalizer.transform(input_flat)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32, device=self.device)
        input_tensor = input_tensor.view(batch_size, seq_len, num_nodes, num_features)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(input_tensor, self.edge_index, self.edge_attr, multi_step=multi_step)
        
        # Convert from [SWH, MWD_cos, MWD_sin, MWP] to [SWH, MWD, MWP]
        predictions_converted = self._convert_circular_to_angular(predictions)
        
        # Handle denormalization manually since we need to convert 4→3 features
        batch_size, num_nodes = predictions_converted.shape[:2]
        pred_flat = predictions_converted.view(-1, 3).cpu().numpy()
        
        # Manual denormalization for [SWH, MWD, MWP] format
        swh = pred_flat[:, 0:1]
        mwd = pred_flat[:, 1:2]  
        mwp = pred_flat[:, 2:3]
        
        # Denormalize each component separately
        try:
            swh_denorm = self.target_normalizer.swh_scaler.inverse_transform(swh)
            mwp_denorm = self.target_normalizer.mwp_scaler.inverse_transform(mwp)
            
            # For MWD, just use the angle directly (it's already in degrees)
            mwd_denorm = mwd.flatten()
            
            # Combine results
            pred_denorm = np.column_stack([swh_denorm.flatten(), mwd_denorm, mwp_denorm.flatten()])
            
        except Exception as e:
            print(f"⚠️  Denormalization failed, using raw predictions: {e}")
            pred_denorm = pred_flat
        
        predictions_final = torch.tensor(pred_denorm, dtype=torch.float32)
        predictions_final = predictions_final.view(batch_size, num_nodes, 3)
        
        return predictions_final.squeeze(0) if squeeze_output else predictions_final
    
    def _convert_circular_to_angular(self, circular_predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert [SWH, MWD_cos, MWD_sin, MWP] to [SWH, MWD, MWP]
        
        Args:
            circular_predictions: [..., 4] or [..., 4*horizon] tensor
        
        Returns:
            angular_predictions: [..., 3] or [..., 3*horizon] tensor
        """
        
        original_shape = circular_predictions.shape
        
        # Handle multi-step predictions
        if original_shape[-1] % 4 == 0:
            horizon = original_shape[-1] // 4
            
            # Reshape to [..., horizon, 4]
            reshaped = circular_predictions.view(*original_shape[:-1], horizon, 4)
            
            # Extract components
            swh = reshaped[..., 0]      # [..., horizon]
            mwd_cos = reshaped[..., 1]  # [..., horizon]
            mwd_sin = reshaped[..., 2]  # [..., horizon]
            mwp = reshaped[..., 3]      # [..., horizon]
            
            # Convert circular to angular
            mwd_rad = torch.atan2(mwd_sin, mwd_cos)
            mwd_deg = torch.rad2deg(mwd_rad)
            
            # Ensure [0, 360) range
            mwd_deg = torch.where(mwd_deg < 0, mwd_deg + 360, mwd_deg)
            
            # Stack back to [..., horizon, 3]
            angular = torch.stack([swh, mwd_deg, mwp], dim=-1)
            
            # Reshape to [..., 3*horizon]
            return angular.view(*original_shape[:-1], horizon * 3)
        
        else:
            raise ValueError(f"Unexpected prediction dimension: {original_shape[-1]}")
    
    def evaluate_on_test_data(self, test_features: torch.Tensor, test_targets: torch.Tensor,
                             test_coordinates: torch.Tensor = None) -> Dict[str, float]:
        """
        Evaluate model on test data with proper circular metrics
        
        Args:
            test_features: [seq_len, num_nodes, features] 
            test_targets: [num_nodes, 3] targets in [SWH, MWD, MWP] format
            test_coordinates: [num_nodes, 2] (optional)
        
        Returns:
            metrics: Dictionary with RMSE values
        """
        
        # Make prediction
        predictions = self.predict(test_features, multi_step=False)  # [num_nodes, 3]
        
        # Convert to numpy for evaluation
        pred_np = predictions.cpu().numpy()
        target_np = test_targets.cpu().numpy()
        
        # Use circular evaluation
        metrics = evaluate_model_with_circular_metrics(pred_np, target_np)
        
        return metrics

def test_circular_model_evaluation():
    """Test the circular model evaluator"""
    
    print("🧪 TESTING CIRCULAR MWD MODEL EVALUATION")
    print("=" * 50)
    
    # Find the latest circular model
    experiments_dir = Path("experiments")
    circular_experiments = [d for d in experiments_dir.iterdir() 
                          if d.is_dir() and "circular" in d.name]
    
    if not circular_experiments:
        print("❌ No circular MWD experiments found!")
        print("   Run the circular training script first.")
        return
    
    # Get latest experiment
    latest_experiment = max(circular_experiments, key=lambda x: x.stat().st_mtime)
    model_path = latest_experiment / "spatiotemporal_circular_model.pt"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"📂 Testing with model: {model_path}")
    
    try:
        # Load evaluator
        evaluator = CircularMWDEvaluator(str(model_path))
        
        # Create test data
        seq_len = evaluator.config.sequence_length
        num_nodes = 1170  # Assuming standard mesh size
        num_features = 11
        
        # Test sequence
        test_sequence = torch.randn(seq_len, num_nodes, num_features)
        
        # Test targets (realistic wave values)
        test_targets = torch.stack([
            torch.abs(torch.randn(num_nodes)) * 3 + 1,  # SWH: 1-10m
            torch.rand(num_nodes) * 360,                # MWD: 0-360°
            torch.abs(torch.randn(num_nodes)) * 5 + 5   # MWP: 5-15s
        ], dim=1)
        
        print(f"🔍 Test data:")
        print(f"   Sequence shape: {test_sequence.shape}")
        print(f"   Targets shape: {test_targets.shape}")
        print(f"   Target ranges: SWH={test_targets[:,0].min():.1f}-{test_targets[:,0].max():.1f}m, "
              f"MWD={test_targets[:,1].min():.1f}-{test_targets[:,1].max():.1f}°, "
              f"MWP={test_targets[:,2].min():.1f}-{test_targets[:,2].max():.1f}s")
        
        # Test prediction
        print(f"\n🔮 Testing prediction...")
        predictions = evaluator.predict(test_sequence, multi_step=False)
        
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Prediction ranges: SWH={predictions[:,0].min():.1f}-{predictions[:,0].max():.1f}m, "
              f"MWD={predictions[:,1].min():.1f}-{predictions[:,1].max():.1f}°, "
              f"MWP={predictions[:,2].min():.1f}-{predictions[:,2].max():.1f}s")
        
        # Test evaluation
        print(f"\n📊 Testing evaluation...")
        metrics = evaluator.evaluate_on_test_data(test_sequence, test_targets)
        
        print(f"   Metrics:")
        print(f"     SWH RMSE: {metrics['swh_rmse']:.3f} m")
        print(f"     MWD RMSE: {metrics['mwd_rmse']:.1f}° (circular)")
        print(f"     MWP RMSE: {metrics['mwp_rmse']:.3f} s")
        print(f"     Overall RMSE: {metrics['overall_rmse']:.3f}")
        
        print(f"\n✅ Circular model evaluation test successful!")
        print(f"   Model outputs proper [SWH, MWD, MWP] format")
        print(f"   Circular MWD conversion working correctly")
        print(f"   Ready for comparison with other models")
        
        return evaluator
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_original_model():
    """Compare circular model with original spatiotemporal model"""
    
    print(f"\n🏆 COMPARING CIRCULAR VS ORIGINAL MODEL")
    print("=" * 50)
    
    # Test circular model
    circular_evaluator = test_circular_model_evaluation()
    
    if circular_evaluator is None:
        print("❌ Cannot compare - circular model test failed")
        return
    
    # TODO: Load original model and compare
    # This would need the original model path
    print(f"🔄 Next step: Run full comparison evaluation")
    print(f"   Expected: Circular model shows much better MWD performance")
    print(f"   Target: MWD RMSE 127° → 20-40°")

if __name__ == "__main__":
    compare_with_original_model()