import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass

from prediction.autoregressive import AutoregressiveWavePredictor

class AutoregressiveEvaluator:
    """
    Evaluates autoregressive forecasts against ground truth
    """
    
    def __init__(self, mesh_loader, feature_names: List[str]):
        self.mesh_loader = mesh_loader
        self.feature_names = feature_names
        self.wave_indices = self._find_wave_indices()
    
    def _find_wave_indices(self):
        """Find wave variable indices"""
        wave_vars = ['swh', 'mwd', 'mwp']
        indices = []
        
        for wave_var in wave_vars:
            for i, feature_name in enumerate(self.feature_names):
                if wave_var in feature_name.lower():
                    indices.append(i)
                    break
        
        return indices
    
    def evaluate_forecast(self, predictions: Dict[int, torch.Tensor], 
                         initial_time_idx: int, time_step_hours: int = 6) -> Dict[str, any]:
        """
        Evaluate autoregressive predictions against ground truth
        
        Args:
            predictions: Output from AutoregressiveWavePredictor.predict_sequence()
            initial_time_idx: Starting time index
            time_step_hours: Hours per timestep
        
        Returns:
            Dictionary with metrics for each forecast horizon
        """
        
        print(f"📊 Evaluating Autoregressive Forecast:")
        print(f"   Horizons: {sorted(predictions.keys())} hours")
        
        evaluation_results = {}
        
        for forecast_hours, pred_waves in predictions.items():
            # Calculate target time index
            forecast_steps = forecast_hours // time_step_hours
            target_time_idx = initial_time_idx + forecast_steps
            
            try:
                # Load ground truth
                target_data = self.mesh_loader.load_features(time_idx=target_time_idx)
                target_features = torch.tensor(target_data['features'], dtype=torch.float32)
                
                # Extract wave variables
                true_waves = target_features[:, self.wave_indices]
                
                # Clean both predictions and targets
                from data.preprocessing import clean_features_for_training
                pred_waves_clean = clean_features_for_training(pred_waves)
                true_waves_clean = clean_features_for_training(true_waves)
                
                # Calculate metrics (reuse existing function)
                from utils.metrics import calculate_wave_metrics
                
                # Reshape for metrics calculation [batch=1, nodes, features]
                pred_batch = pred_waves_clean.unsqueeze(0)
                true_batch = true_waves_clean.unsqueeze(0)
                
                metrics = calculate_wave_metrics(pred_batch, true_batch)
                
                # Add some extra statistics
                metrics.update({
                    'forecast_hours': forecast_hours,
                    'target_time_idx': target_time_idx,
                    'num_nodes': len(pred_waves),
                    'pred_swh_mean': pred_waves[:, 0].mean().item(),
                    'true_swh_mean': true_waves[:, 0].mean().item(),
                    'swh_bias': (pred_waves[:, 0].mean() - true_waves[:, 0].mean()).item()
                })
                
                evaluation_results[forecast_hours] = metrics
                
                print(f"   t+{forecast_hours:2d}h: RMSE={metrics.get('swh_rmse', 0):.4f}m, "
                      f"Bias={metrics['swh_bias']:+.3f}m, "
                      f"Corr={metrics.get('swh_corr', 0):.3f}")
                
            except Exception as e:
                print(f"   ⚠️  t+{forecast_hours}h: Evaluation failed: {e}")
                evaluation_results[forecast_hours] = {'error': str(e)}
        
        return evaluation_results
    
    def evaluate_multiple_forecasts(self, predictor: AutoregressiveWavePredictor,
                                   test_time_indices: List[int],
                                   forecast_steps: List[int] = [1, 2, 4, 8, 12],
                                   time_step_hours: int = 6) -> Dict[str, any]:
        """
        Evaluate autoregressive performance across multiple starting times
        """
        
        print(f"🧪 Multi-Forecast Evaluation:")
        print(f"   Test times: {len(test_time_indices)}")
        print(f"   Forecast horizons: {[s * time_step_hours for s in forecast_steps]} hours")
        
        all_results = {}
        forecast_hours_list = [s * time_step_hours for s in forecast_steps]
        
        # Initialize storage for aggregated metrics
        for hours in forecast_hours_list:
            all_results[hours] = {
                'swh_rmse': [], 'swh_mae': [], 'swh_corr': [], 'swh_bias': [],
                'mwd_rmse': [], 'mwd_mae': [],
                'mwp_rmse': [], 'mwp_mae': []
            }
        
        # Run forecasts for each test time
        for i, test_time in enumerate(test_time_indices):
            print(f"\n--- Forecast {i+1}/{len(test_time_indices)} (t={test_time}) ---")
            
            try:
                # Make prediction
                predictions = predictor.predict_sequence(
                    initial_time_idx=test_time,
                    forecast_steps=forecast_steps,
                    time_step_hours=time_step_hours
                )
                
                # Evaluate this forecast
                forecast_results = self.evaluate_forecast(predictions, test_time, time_step_hours)
                
                # Aggregate results
                for hours, metrics in forecast_results.items():
                    if 'error' not in metrics:
                        for metric_name in all_results[hours].keys():
                            if metric_name in metrics:
                                all_results[hours][metric_name].append(metrics[metric_name])
                
            except Exception as e:
                print(f"   ❌ Forecast failed: {e}")
                continue
        
        # Calculate summary statistics
        summary_results = {}
        for hours in forecast_hours_list:
            metrics = all_results[hours]
            summary_results[hours] = {}
            
            for metric_name, values in metrics.items():
                if values:
                    summary_results[hours][f'{metric_name}_mean'] = np.mean(values)
                    summary_results[hours][f'{metric_name}_std'] = np.std(values)
                    summary_results[hours][f'{metric_name}_count'] = len(values)
        
        # Print summary
        print(f"\n📊 AUTOREGRESSIVE PERFORMANCE SUMMARY:")
        print("=" * 60)
        print("Horizon  | SWH RMSE | SWH Bias | SWH Corr | Tests")
        print("-" * 60)
        
        for hours in sorted(forecast_hours_list):
            results = summary_results.get(hours, {})
            rmse = results.get('swh_rmse_mean', 0)
            bias = results.get('swh_bias_mean', 0)
            corr = results.get('swh_corr_mean', 0)
            count = results.get('swh_rmse_count', 0)
            
            print(f"t+{hours:2d}h    | {rmse:.4f}m  | {bias:+.3f}m  | {corr:.3f}   | {count}")
        
        return summary_results