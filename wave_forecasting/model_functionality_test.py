#!/usr/bin/env python3
"""
Deep diagnostic to check if the model is actually functioning
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

def test_model_basic_functionality():
    """Test if the model is actually working or just outputting constants"""
    
    print("🔍 TESTING MODEL BASIC FUNCTIONALITY")
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
    latest_checkpoint = all_checkpoints[0][1]
    
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
    print(f"📦 Model loaded: {predictor.experiment_id}")
    print(f"   Parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
    
    # TEST 1: Different inputs should give different outputs
    print(f"\n🧪 TEST 1: Input Sensitivity")
    print("-" * 25)
    
    # Create test inputs
    num_nodes = 100
    num_features = 11
    
    # Input 1: All zeros
    input1 = torch.zeros(num_nodes, num_features)
    
    # Input 2: All ones
    input2 = torch.ones(num_nodes, num_features)
    
    # Input 3: Random
    input3 = torch.randn(num_nodes, num_features)
    
    # Input 4: Random (different)
    input4 = torch.randn(num_nodes, num_features) * 5
    
    # Make predictions
    with torch.no_grad():
        pred1 = predictor.predict(input1)
        pred2 = predictor.predict(input2)
        pred3 = predictor.predict(input3)
        pred4 = predictor.predict(input4)
    
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
        return False
    else:
        print(f"   ✅ Model responds to different inputs")
    
    # TEST 2: Check if gradients flow
    print(f"\n🧪 TEST 2: Gradient Flow")
    print("-" * 20)
    
    input_test = torch.randn(num_nodes, num_features, requires_grad=True)
    
    # Forward pass
    output = predictor.model(input_test, predictor.edge_index, predictor.edge_attr)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    if input_test.grad is not None:
        grad_norm = input_test.grad.norm()
        print(f"   Input gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 1e-6:
            print(f"   ✅ Gradients are flowing")
        else:
            print(f"   ❌ PROBLEM: Gradients are zero!")
            return False
    else:
        print(f"   ❌ PROBLEM: No gradients computed!")
        return False
    
    # TEST 3: Check individual model components
    print(f"\n🧪 TEST 3: Model Component Analysis")
    print("-" * 30)
    
    model = predictor.model
    
    # Check encoder
    test_input = torch.randn(num_nodes, num_features)
    
    print(f"   Testing encoder...")
    with torch.no_grad():
        if hasattr(model, 'encoder'):
            encoded = model.encoder(test_input)
            print(f"     Encoder output: {encoded.shape}, range {encoded.min():.3f} to {encoded.max():.3f}")
            
            # Check if encoder weights are reasonable
            first_layer = model.encoder[0]
            if hasattr(first_layer, 'weight'):
                weight_norm = first_layer.weight.norm()
                print(f"     First layer weight norm: {weight_norm:.3f}")
                
                if weight_norm < 1e-6:
                    print(f"     ❌ PROBLEM: Encoder weights are near zero!")
                    return False
        else:
            print(f"     ❌ No encoder found!")
            return False
    
    # Check message passing layers
    print(f"   Testing message passing...")
    if hasattr(model, 'message_layers'):
        with torch.no_grad():
            h = encoded.clone()
            
            for i, layer in enumerate(model.message_layers):
                h_before = h.clone()
                try:
                    h_new = layer(h, predictor.edge_index, predictor.edge_attr)
                    h = h + h_new  # Residual connection
                    
                    h_change = torch.mean(torch.abs(h - h_before))
                    print(f"     Layer {i}: Change = {h_change:.6f}")
                    
                    if h_change < 1e-8:
                        print(f"     ⚠️  Layer {i} produces minimal change")
                    
                except Exception as e:
                    print(f"     ❌ Layer {i} failed: {e}")
                    return False
    else:
        print(f"     ❌ No message passing layers found!")
        return False
    
    # Check decoder
    print(f"   Testing decoder...")
    if hasattr(model, 'decoder'):
        with torch.no_grad():
            decoded = model.decoder(h)
            print(f"     Decoder output: {decoded.shape}, range {decoded.min():.3f} to {decoded.max():.3f}")
    else:
        print(f"     ❌ No decoder found!")
        return False
    
    return True

def test_edge_connectivity():
    """Test if edge connectivity is working properly"""
    
    print(f"\n🔍 TESTING EDGE CONNECTIVITY")
    print("-" * 30)
    
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
    latest_checkpoint = all_checkpoints[0][1]
    
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    
    edge_index = predictor.edge_index
    edge_attr = predictor.edge_attr
    
    print(f"   Edge index shape: {edge_index.shape}")
    print(f"   Edge attr shape: {edge_attr.shape}")
    
    # Check edge index validity
    max_node_idx = edge_index.max().item()
    min_node_idx = edge_index.min().item()
    
    print(f"   Node indices range: {min_node_idx} to {max_node_idx}")
    
    # Test with a small graph first
    num_test_nodes = max_node_idx + 1
    test_features = torch.randn(num_test_nodes, 11)
    
    try:
        with torch.no_grad():
            test_output = predictor.predict(test_features)
        
        print(f"   ✅ Edge connectivity works with {num_test_nodes} nodes")
        print(f"   Output shape: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Edge connectivity failed: {e}")
        return False

def test_model_weights_loaded():
    """Check if model weights were actually loaded from checkpoint"""
    
    print(f"\n🔍 TESTING MODEL WEIGHT LOADING")
    print("-" * 35)
    
    from prediction.forecasting import WavePredictor
    from models.spatial import SpatialWaveGNN
    from config.base import ModelConfig
    
    # Load the actual model
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
    
    # Load checkpoint directly
    checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    print(f"   Checkpoint keys: {len(state_dict.keys())}")
    
    # Check some specific weights
    key_weights = [
        'encoder.0.weight',
        'encoder.0.bias', 
        'message_layers.0.linear.weight',
        'decoder.0.weight'
    ]
    
    for key in key_weights:
        if key in state_dict:
            weight = state_dict[key]
            weight_norm = torch.norm(weight)
            weight_mean = torch.mean(weight)
            
            print(f"   {key}: norm={weight_norm:.3f}, mean={weight_mean:.6f}")
            
            if weight_norm < 1e-6:
                print(f"     ❌ PROBLEM: {key} has near-zero weights!")
                return False
        else:
            print(f"   ⚠️  Key {key} not found in checkpoint")
    
    # Create a fresh model and compare
    predictor = WavePredictor.from_checkpoint(latest_checkpoint)
    loaded_model = predictor.model
    
    # Check if the loaded model has the same weights
    for name, param in loaded_model.named_parameters():
        if name in state_dict:
            original_weight = state_dict[name]
            loaded_weight = param.data
            
            diff = torch.mean(torch.abs(original_weight - loaded_weight))
            
            if diff < 1e-8:
                print(f"   ✅ {name}: weights loaded correctly")
            else:
                print(f"   ❌ {name}: weight loading error! Diff={diff:.6f}")
                return False
    
    return True

def main():
    """Main diagnostic function"""
    
    print("🔧 MODEL FUNCTIONALITY DEEP DIAGNOSTIC")
    print("=" * 50)
    
    try:
        # Test basic functionality
        basic_ok = test_model_basic_functionality()
        
        if not basic_ok:
            print(f"\n❌ BASIC FUNCTIONALITY FAILED")
            print(f"   Model is not responding to inputs properly")
            return
        
        # Test edge connectivity
        edge_ok = test_edge_connectivity()
        
        if not edge_ok:
            print(f"\n❌ EDGE CONNECTIVITY FAILED")
            print(f"   Graph neural network components not working")
            return
        
        # Test weight loading
        weights_ok = test_model_weights_loaded()
        
        if not weights_ok:
            print(f"\n❌ WEIGHT LOADING FAILED")
            print(f"   Model weights were not loaded correctly from checkpoint")
            return
        
        print(f"\n✅ ALL DIAGNOSTICS PASSED")
        print(f"   Model appears to be functioning correctly")
        print(f"   The 1.04m RMSE issue is likely in:")
        print(f"   - Data preprocessing differences")
        print(f"   - Evaluation methodology")
        print(f"   - Training vs evaluation data distribution")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()