#!/usr/bin/env python3
"""
Compare Circular MWD Model vs Original Spatiotemporal Model
Real evaluation on the same test data to measure improvement
"""

import sys
import torch
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path.cwd()))

# Import evaluators
from circular_mwd_evaluator import CircularMWDEvaluator
from new_architecture_test import SpatioTemporalEvaluator, SpatioTemporalConfig  # Original evaluator
from mwd_circular_fixes import evaluate_model_with_circular_metrics

def find_models():
    """Find the latest circular and original models"""
    
    experiments_dir = Path("experiments")
    
    # Find circular model
    circular_experiments = [d for d in experiments_dir.iterdir() 
                          if d.is_dir() and "circular" in d.name]
    
    if not circular_experiments:
        print("❌ No circular MWD experiments found!")
        return None, None
    
    latest_circular = max(circular_experiments, key=lambda x: x.stat().st_mtime)
    circular_model_path = latest_circular / "spatiotemporal_circular_model.pt"
    
    # Find original spatiotemporal model
    spatiotemporal_experiments = [d for d in experiments_dir.iterdir() 
                                if d.is_dir() and "spatiotemporal" in d.name and "circular" not in d.name]
    
    if not spatiotemporal_experiments:
        print("❌ No original spatiotemporal experiments found!")
        return circular_model_path, None
    
    latest_original = max(spatiotemporal_experiments, key=lambda x: x.stat().st_mtime)
    original_model_path = latest_original / "spatiotemporal_model.pt"
    
    return circular_model_path, original_model_path

def load_real_test_data():
    """Load real ERA5 test data for evaluation"""
    
    print("📊 Loading real test data...")
    
    try:
        from data.loaders import ERA5DataManager, GEBCODataManager
        from data.preprocessing import MultiResolutionInterpolator
        from data.datasets import MeshDataLoader
        from mesh.icosahedral import IcosahedralMesh
        from config.base import DataConfig, MeshConfig
        
        # Load data components
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load February 2020 data (different from training month)
        era5_atmo, era5_waves = era5_manager.load_month_data(2020, 2)
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Get coordinates
        lat, lon = mesh.vertices_to_lat_lon()
        regional_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        coordinates = torch.tensor(
            np.column_stack([lat[regional_indices], lon[regional_indices]]), 
            dtype=torch.float32
        )
        
        # Create test sequences
        test_sequences = []
        test_targets = []
        
        sequence_length = 6
        max_samples = 20  # Reasonable number for comparison
        
        for t in range(0, min(50, len(era5_atmo.valid_time)), 3):  # Every 3rd timestep
            try:
                # Build sequence
                sequence_features = []
                for i in range(sequence_length):
                    if t + i < len(era5_atmo.valid_time):
                        features_data = mesh_loader.load_features(time_idx=t + i)
                        features = torch.tensor(features_data['features'], dtype=torch.float32)
                        features = torch.nan_to_num(features, nan=0.0)
                        sequence_features.append(features)
                
                # Get target
                if t + sequence_length < len(era5_atmo.valid_time):
                    target_data = mesh_loader.load_features(time_idx=t + sequence_length)
                    targets = torch.tensor(target_data['features'][:, [3, 4, 5]], dtype=torch.float32)  # SWH, MWD, MWP
                    targets = torch.nan_to_num(targets, nan=0.0)
                    
                    if len(sequence_features) == sequence_length:
                        sequence_tensor = torch.stack(sequence_features, dim=0)  # [6, num_nodes, features]
                        test_sequences.append(sequence_tensor)
                        test_targets.append(targets)
                        
                        if len(test_sequences) >= max_samples:
                            break
                
            except Exception as e:
                print(f"   ⚠️  Skipping timestep {t}: {e}")
                continue
        
        print(f"   ✅ Loaded {len(test_sequences)} test sequences")
        print(f"   📊 Sequence shape: {test_sequences[0].shape}")
        print(f"   📊 Target shape: {test_targets[0].shape}")
        
        return test_sequences, test_targets, coordinates
        
    except Exception as e:
        print(f"   ❌ Failed to load real data: {e}")
        return None, None, None

def evaluate_model_performance(evaluator, test_sequences, test_targets, model_name):
    """Evaluate a model on test data"""
    
    print(f"\n🔍 Evaluating {model_name}...")
    
    all_metrics = []
    all_predictions = []
    all_targets = []
    
    for i, (sequence, targets) in enumerate(zip(test_sequences, test_targets)):
        if i % 5 == 0:
            print(f"   Sample {i+1}/{len(test_sequences)}")
        
        try:
            # Make prediction
            if hasattr(evaluator, 'evaluate_on_test_data'):
                # Circular model
                metrics = evaluator.evaluate_on_test_data(sequence, targets)
                predictions = evaluator.predict(sequence, multi_step=False)
            else:
                # Original model
                predictions = evaluator.predict(sequence.unsqueeze(0), multi_step=False)
                predictions = predictions.squeeze(0)
                
                # Calculate metrics manually
                pred_np = predictions.cpu().numpy()
                target_np = targets.cpu().numpy()
                metrics = evaluate_model_with_circular_metrics(pred_np, target_np)
            
            all_metrics.append(metrics)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
        except Exception as e:
            print(f"     ⚠️  Failed on sample {i}: {e}")
            continue
    
    if not all_metrics:
        print(f"   ❌ No successful evaluations for {model_name}")
        return None
    
    # Aggregate metrics
    avg_metrics = {
        'swh_rmse': np.mean([m['swh_rmse'] for m in all_metrics]),
        'mwd_rmse': np.mean([m['mwd_rmse'] for m in all_metrics]),
        'mwp_rmse': np.mean([m['mwp_rmse'] for m in all_metrics]),
        'overall_rmse': np.mean([m['overall_rmse'] for m in all_metrics])
    }
    
    # Standard deviations
    std_metrics = {
        'swh_rmse_std': np.std([m['swh_rmse'] for m in all_metrics]),
        'mwd_rmse_std': np.std([m['mwd_rmse'] for m in all_metrics]),
        'mwp_rmse_std': np.std([m['mwp_rmse'] for m in all_metrics]),
        'overall_rmse_std': np.std([m['overall_rmse'] for m in all_metrics])
    }
    
    print(f"   ✅ {model_name} evaluation complete:")
    print(f"     SWH RMSE: {avg_metrics['swh_rmse']:.3f} ± {std_metrics['swh_rmse_std']:.3f} m")
    print(f"     MWD RMSE: {avg_metrics['mwd_rmse']:.1f} ± {std_metrics['mwd_rmse_std']:.1f}°")
    print(f"     MWP RMSE: {avg_metrics['mwp_rmse']:.3f} ± {std_metrics['mwp_rmse_std']:.3f} s")
    print(f"     Overall: {avg_metrics['overall_rmse']:.3f} ± {std_metrics['overall_rmse_std']:.3f}")
    
    return {**avg_metrics, **std_metrics, 'num_samples': len(all_metrics)}

def main():
    """Main comparison function"""
    
    print("🏆 CIRCULAR MWD vs ORIGINAL MODEL COMPARISON")
    print("=" * 60)
    
    # Find models
    circular_path, original_path = find_models()
    
    if circular_path is None:
        print("❌ Cannot run comparison - no circular model found")
        return
    
    if original_path is None:
        print("❌ Cannot run comparison - no original model found")
        return
    
    print(f"📂 Models found:")
    print(f"   Circular: {circular_path}")
    print(f"   Original: {original_path}")
    
    # Load real test data
    test_sequences, test_targets, coordinates = load_real_test_data()
    
    if test_sequences is None:
        print("❌ Cannot run comparison - failed to load test data")
        return
    
    # Load evaluators
    print(f"\n🔧 Loading models...")
    
    try:
        circular_evaluator = CircularMWDEvaluator(str(circular_path))
        print(f"   ✅ Circular model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load circular model: {e}")
        return
    
    try:
        original_evaluator = SpatioTemporalEvaluator(str(original_path))
        print(f"   ✅ Original model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load original model: {e}")
        return
    
    # Run evaluations
    start_time = time.time()
    
    circular_results = evaluate_model_performance(
        circular_evaluator, test_sequences, test_targets, "Circular MWD Model"
    )
    
    original_results = evaluate_model_performance(
        original_evaluator, test_sequences, test_targets, "Original Model"
    )
    
    evaluation_time = time.time() - start_time
    
    # Compare results
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 40)
    
    if circular_results and original_results:
        print(f"{'Metric':<15} {'Original':<15} {'Circular':<15} {'Improvement':<15}")
        print("-" * 60)
        
        # SWH comparison
        swh_improvement = ((original_results['swh_rmse'] - circular_results['swh_rmse']) / 
                          original_results['swh_rmse'] * 100)
        print(f"{'SWH RMSE (m)':<15} {original_results['swh_rmse']:<15.3f} "
              f"{circular_results['swh_rmse']:<15.3f} {swh_improvement:<15.1f}%")
        
        # MWD comparison (the key metric!)
        mwd_improvement = ((original_results['mwd_rmse'] - circular_results['mwd_rmse']) / 
                          original_results['mwd_rmse'] * 100)
        print(f"{'MWD RMSE (°)':<15} {original_results['mwd_rmse']:<15.1f} "
              f"{circular_results['mwd_rmse']:<15.1f} {mwd_improvement:<15.1f}%")
        
        # MWP comparison
        mwp_improvement = ((original_results['mwp_rmse'] - circular_results['mwp_rmse']) / 
                          original_results['mwp_rmse'] * 100)
        print(f"{'MWP RMSE (s)':<15} {original_results['mwp_rmse']:<15.3f} "
              f"{circular_results['mwp_rmse']:<15.3f} {mwp_improvement:<15.1f}%")
        
        # Overall comparison
        overall_improvement = ((original_results['overall_rmse'] - circular_results['overall_rmse']) / 
                              original_results['overall_rmse'] * 100)
        print(f"{'Overall RMSE':<15} {original_results['overall_rmse']:<15.3f} "
              f"{circular_results['overall_rmse']:<15.3f} {overall_improvement:<15.1f}%")
        
        print(f"\n🎯 KEY RESULTS:")
        print(f"   MWD RMSE: {original_results['mwd_rmse']:.1f}° → {circular_results['mwd_rmse']:.1f}° "
              f"({mwd_improvement:+.1f}%)")
        print(f"   Overall RMSE: {original_results['overall_rmse']:.3f} → {circular_results['overall_rmse']:.3f} "
              f"({overall_improvement:+.1f}%)")
        
        # Success assessment
        print(f"\n🏆 ASSESSMENT:")
        if mwd_improvement > 50:
            print(f"   🎉 MAJOR SUCCESS: >50% MWD improvement!")
        elif mwd_improvement > 20:
            print(f"   ✅ SUCCESS: Significant MWD improvement")
        elif mwd_improvement > 0:
            print(f"   📈 IMPROVEMENT: Modest MWD improvement")
        else:
            print(f"   ❌ NO IMPROVEMENT: Circular MWD did not help")
        
        # Target assessment
        if circular_results['overall_rmse'] < 5.0:
            print(f"   🎯 CLOSE TO TARGET: Overall RMSE < 5.0")
        elif circular_results['overall_rmse'] < 10.0:
            print(f"   🎯 PROGRESS: Overall RMSE < 10.0")
        
        print(f"\n⏱️  Evaluation completed in {evaluation_time:.1f} seconds")
        print(f"📊 Tested on {circular_results['num_samples']} real wave forecast samples")
        
    else:
        print("❌ Comparison failed - missing results")

if __name__ == "__main__":
    main()