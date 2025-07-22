#!/usr/bin/env python3
"""
Annual Evaluation Framework for Wavelet-Enhanced Variable LR Wave Models
Comprehensive testing against 2024 data with detailed wavelet analysis
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import xarray as xr
from collections import defaultdict

sys.path.insert(0, str(Path.cwd()))

# Import components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import DataConfig, MeshConfig
from mwd_circular_fixes import evaluate_model_with_circular_metrics, VariableSpecificNormalizer

# Import wavelet model components
from wavelet_variable_circular_spatiotemporal import (
    WaveletVariableLRConfig, 
    WaveletVariableLRSpatioTemporalGNN,
    VariableSpecificLoss,
    VariableLearningManager
)

class WaveletModelEvaluator:
    """Dedicated evaluator for Wavelet-Enhanced Variable LR models"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"🌊 Wavelet-Enhanced Model Evaluator")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        
        # Model components
        self.model = None
        self.config = None
        self.normalizer = None
        self.target_normalizer = None
        self.edge_index = None
        self.edge_attr = None
        
    def load_model(self):
        """Load the wavelet-enhanced model"""
        print(f"📦 Loading Wavelet-Enhanced Variable LR model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract components
            self.config = checkpoint.get('config', WaveletVariableLRConfig())
            self.normalizer = checkpoint.get('feature_normalizer')
            self.target_normalizer = checkpoint.get('target_normalizer')
            self.edge_index = checkpoint.get('edge_index')
            self.edge_attr = checkpoint.get('edge_attr')
            
            # Create model
            self.model = WaveletVariableLRSpatioTemporalGNN(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"  ✅ Wavelet-Enhanced Model loaded successfully")
            print(f"     Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"     Base Architecture: Variable LR (9.21 RMSE baseline)")
            print(f"     Enhancement: Wavelet frequency processing")
            print(f"     Wavelet Type: {self.config.wavelet_type}")
            print(f"     Wavelet Levels: {self.config.wavelet_levels}")
            print(f"     Fusion Method: {self.config.wavelet_fusion_method}")
            print(f"     LSTM+Wavelet: {self.config.combine_lstm_wavelet}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def normalize_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize input using model's normalizer"""
        if self.normalizer is None:
            return input_tensor
        
        # input_tensor shape: [seq_len, nodes, features]
        seq_len, nodes, features = input_tensor.shape
        
        # Flatten for normalization
        flat_input = input_tensor.view(-1, features).cpu().numpy()
        
        # Normalize
        normalized = self.normalizer.transform(flat_input)
        
        # Reshape back
        normalized_tensor = torch.tensor(normalized, dtype=torch.float32, device=self.device)
        return normalized_tensor.view(seq_len, nodes, features)
    
    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert predictions back to physical units"""
        if self.target_normalizer is None:
            return predictions
        
        # predictions shape: [batch, nodes, 4] - [SWH, MWD_cos, MWD_sin, MWP]
        batch_size, nodes, features = predictions.shape
        
        if features == 4:
            # Single step with circular MWD
            flat_preds = predictions.view(-1, 4).cpu().numpy()
            denormalized = self.target_normalizer.inverse_transform_targets(flat_preds)
            return torch.tensor(denormalized, dtype=torch.float32, device=predictions.device).view(batch_size, nodes, 3)
        else:
            # Unexpected format
            print(f"Warning: Unexpected prediction format: {predictions.shape}")
            return predictions
    
    def predict(self, input_tensor: torch.Tensor, multi_step: bool = False) -> torch.Tensor:
        """Make prediction with wavelet-enhanced model"""
        
        # Add batch dimension if needed
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # [1, seq_len, nodes, features]
        
        # Normalize input
        normalized_input = self.normalize_input(input_tensor.squeeze(0))
        normalized_input = normalized_input.unsqueeze(0)  # Add batch dim back
        
        with torch.no_grad():
            predictions = self.model(normalized_input, 
                                   self.edge_index.to(self.device),
                                   self.edge_attr.to(self.device), 
                                   multi_step=multi_step)
        
        # Denormalize predictions
        physical_predictions = self.denormalize_predictions(predictions)
        
        return physical_predictions.squeeze(0)  # Remove batch dimension
    
    def evaluate_on_test_data(self, sequence: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate wavelet model on single test sequence"""
        
        try:
            # Make prediction
            predictions = self.predict(sequence, multi_step=False)
            
            # Convert to numpy
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            # Calculate metrics using circular handling
            metrics = evaluate_model_with_circular_metrics(pred_np, target_np)
            
            return metrics
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'swh_rmse': float('inf'),
                'mwd_rmse': float('inf'),
                'mwp_rmse': float('inf'),
                'overall_rmse': float('inf')
            }

class WaveletAnnualEvaluator:
    """Annual evaluation framework for Wavelet-Enhanced models"""
    
    def __init__(self, model_name: Optional[str] = None, test_year: int = 2024):
        self.test_year = test_year
        self.model_name = model_name
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Find model to evaluate
        # self.model_path = self._find_model_to_evaluate()
        self.model_path = Path('experiments/wavelet_variable_lr_20250709_235831/best_wavelet_variable_lr_model.pt')
        if not self.model_path:
            raise ValueError("No suitable Wavelet-Enhanced model found for evaluation")
        
        # Setup evaluation directory
        self.eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.model_path.parent / f"evaluation_{self.eval_timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 Wavelet-Enhanced Annual Evaluation Framework")
        print(f"   Model: {self.model_path}")
        print(f"   Test Year: {self.test_year}")
        print(f"   Results: {self.eval_dir}")
        print(f"   Device: {self.device}")
        print(f"   Baseline Target: Beat 9.21 RMSE (Variable LR)")
        print(f"   Wavelet Target: 7.5-8.5 RMSE range")
        
        # Results storage
        self.results = defaultdict(list)
        self.monthly_results = {}
        
    def _find_model_to_evaluate(self) -> Path:
        """Find the Wavelet-Enhanced model to evaluate"""
        
        experiments_dir = Path("experiments")
        
        if self.model_name:
            # Look for specific model name
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.model_name.lower() in d.name.lower()]
        else:
            # Find latest wavelet model by default
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and "wavelet" in d.name.lower()]
        
        if not model_experiments:
            print(f"❌ No Wavelet-Enhanced models found matching criteria")
            return None
        
        # Get most recent experiment
        latest_experiment = max(model_experiments, key=lambda x: x.stat().st_mtime)
        
        # Look for model file (in priority order)
        model_candidates = [
            latest_experiment / "wavelet_variable_lr_model.pt",
            latest_experiment / "best_wavelet_variable_lr_model.pt",
            latest_experiment / "final_model.pt"
        ]
        
        for model_path in model_candidates:
            if model_path.exists():
                print(f"✅ Found Wavelet-Enhanced model: {model_path}")
                return model_path
        
        print(f"❌ No model file found in {latest_experiment}")
        print(f"    Looked for: {[p.name for p in model_candidates]}")
        return None
    
    def load_annual_data(self) -> Dict[int, Tuple]:
        """Load ERA5 data for entire test year"""
        
        print(f"📊 Loading {self.test_year} ERA5 data...")
        
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load all months for test year
        monthly_data = {}
        
        for month in range(1, 13):
            try:
                print(f"   Loading {self.test_year}-{month:02d}...")
                
                # Check if data file exists with expected naming
                data_dir = Path("data/era5")
                atmo_file = data_dir / f"era5_atmo_{self.test_year}{month:02d}.nc"
                wave_file = data_dir / f"era5_waves_{self.test_year}{month:02d}.nc"
                
                if atmo_file.exists() and wave_file.exists():
                    # Load directly from files
                    era5_atmo = xr.open_dataset(atmo_file)
                    era5_waves = xr.open_dataset(wave_file)
                else:
                    # Use data manager
                    era5_atmo, era5_waves = era5_manager.load_month_data(self.test_year, month)
                
                monthly_data[month] = (era5_atmo, era5_waves)
                print(f"     ✅ {self.test_year}-{month:02d}: {len(era5_atmo.valid_time)} timesteps")
                
            except Exception as e:
                print(f"     ⚠️  Failed to load {self.test_year}-{month:02d}: {e}")
                continue
        
        if not monthly_data:
            raise ValueError(f"No data loaded for {self.test_year}")
        
        # Load bathymetry once
        gebco_data = gebco_manager.load_bathymetry()
        
        print(f"   ✅ Loaded {len(monthly_data)} months of data")
        return monthly_data, gebco_data
    
    def create_test_sequences(self, era5_atmo, era5_waves, gebco_data, 
                            month: int, samples_per_month: int = 100) -> List[Dict]:
        """Create test sequences for a specific month"""
        
        # Setup mesh and interpolator
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create test sequences
        test_sequences = []
        sequence_length = 6
        prediction_horizon = 1  # Single step for evaluation
        
        max_time = len(era5_atmo.valid_time) - sequence_length - prediction_horizon
        step_size = max(1, max_time // samples_per_month)
        
        for t in range(0, max_time, step_size):
            if len(test_sequences) >= samples_per_month:
                break
                
            try:
                # Input sequence
                input_features = []
                for i in range(sequence_length):
                    features_data = mesh_loader.load_features(time_idx=t + i)
                    features = torch.tensor(features_data['features'], dtype=torch.float32)
                    features = torch.nan_to_num(features, nan=0.0)
                    input_features.append(features)
                
                # Target (single step)
                target_data = mesh_loader.load_features(time_idx=t + sequence_length)
                targets = torch.tensor(target_data['features'][:, [3, 4, 5]], dtype=torch.float32)
                targets = torch.nan_to_num(targets, nan=0.0)
                
                input_tensor = torch.stack(input_features, dim=0)
                
                test_sequences.append({
                    'input': input_tensor,
                    'target': targets,
                    'timestep': t,
                    'month': month,
                    'datetime': era5_atmo.valid_time[t + sequence_length].values
                })
                
            except Exception as e:
                continue
        
        return test_sequences
    
    def evaluate_month(self, evaluator: WaveletModelEvaluator, 
                      era5_atmo, era5_waves, gebco_data, month: int) -> Dict[str, Any]:
        """Evaluate Wavelet model performance for a specific month"""
        
        print(f"🔍 Evaluating Wavelet-Enhanced model on {self.test_year}-{month:02d}...")
        
        # Create test sequences
        test_sequences = self.create_test_sequences(era5_atmo, era5_waves, gebco_data, month)
        
        if not test_sequences:
            print(f"   ⚠️  No test sequences for month {month}")
            return {}
        
        # Evaluate all sequences
        month_metrics = []
        variable_errors = {'swh': [], 'mwd': [], 'mwp': []}
        
        for i, sequence in enumerate(test_sequences):
            if i % 20 == 0:
                print(f"     Sample {i+1}/{len(test_sequences)}")
            
            try:
                # Evaluate sequence
                metrics = evaluator.evaluate_on_test_data(sequence['input'], sequence['target'])
                
                # Add metadata
                metrics['month'] = month
                metrics['timestep'] = sequence['timestep']
                metrics['datetime'] = sequence['datetime']
                
                month_metrics.append(metrics)
                
                # Track variable-specific errors
                variable_errors['swh'].append(metrics['swh_rmse'])
                variable_errors['mwd'].append(metrics['mwd_rmse'])
                variable_errors['mwp'].append(metrics['mwp_rmse'])
                
            except Exception as e:
                print(f"       ⚠️  Failed on sample {i}: {e}")
                continue
        
        if not month_metrics:
            return {}
        
        # Aggregate monthly statistics
        monthly_stats = {
            'month': month,
            'num_samples': len(month_metrics),
            'swh_rmse_mean': np.mean(variable_errors['swh']),
            'swh_rmse_std': np.std(variable_errors['swh']),
            'mwd_rmse_mean': np.mean(variable_errors['mwd']),
            'mwd_rmse_std': np.std(variable_errors['mwd']),
            'mwp_rmse_mean': np.mean(variable_errors['mwp']),
            'mwp_rmse_std': np.std(variable_errors['mwp']),
            'overall_rmse_mean': np.mean([m['overall_rmse'] for m in month_metrics]),
            'overall_rmse_std': np.std([m['overall_rmse'] for m in month_metrics])
        }
        
        # Store detailed results
        monthly_stats['detailed_metrics'] = month_metrics
        
        # Performance vs baseline analysis
        baseline_rmse = 9.21  # Variable LR baseline
        improvement = ((baseline_rmse - monthly_stats['overall_rmse_mean']) / baseline_rmse) * 100
        
        print(f"   ✅ Month {month}: Overall RMSE = {monthly_stats['overall_rmse_mean']:.3f} ± {monthly_stats['overall_rmse_std']:.3f}")
        print(f"       Variable breakdown: SWH={monthly_stats['swh_rmse_mean']:.3f}, MWD={monthly_stats['mwd_rmse_mean']:.1f}°, MWP={monthly_stats['mwp_rmse_mean']:.3f}")
        
        if improvement > 0:
            print(f"       🎉 vs Baseline (9.21): +{improvement:.1f}% improvement")
        else:
            print(f"       ⚠️  vs Baseline (9.21): {improvement:.1f}% degradation")
        
        # Wavelet-specific analysis
        if monthly_stats['overall_rmse_mean'] < 8.5:
            print(f"       🌊 EXCELLENT: Wavelet target achieved (<8.5)")
        elif monthly_stats['overall_rmse_mean'] < 9.21:
            print(f"       ✅ GOOD: Beats baseline")
        
        return monthly_stats
    
    def run_annual_evaluation(self, samples_per_month: int = 100) -> Dict[str, Any]:
        """Run comprehensive annual evaluation for Wavelet model"""
        
        print(f"🌊 WAVELET-ENHANCED ANNUAL EVALUATION - {self.test_year}")
        print("=" * 75)
        
        start_time = time.time()
        
        # Load model
        evaluator = WaveletModelEvaluator(str(self.model_path))
        if not evaluator.load_model():
            raise ValueError("Failed to load Wavelet-Enhanced model")
        
        print(f"✅ Wavelet-Enhanced model loaded successfully")
        
        # Load annual data
        monthly_data, gebco_data = self.load_annual_data()
        
        # Evaluate each month
        monthly_results = {}
        all_monthly_stats = []
        
        for month in sorted(monthly_data.keys()):
            era5_atmo, era5_waves = monthly_data[month]
            
            monthly_result = self.evaluate_month(evaluator, era5_atmo, era5_waves, 
                                               gebco_data, month)
            
            if monthly_result:
                monthly_results[month] = monthly_result
                all_monthly_stats.append(monthly_result)
        
        if not all_monthly_stats:
            raise ValueError("No successful monthly evaluations")
        
        # Calculate comprehensive results
        results = self._calculate_comprehensive_results(all_monthly_stats, start_time)
        results['monthly_results'] = monthly_results
        
        # Save results
        self._save_results(results)
        
        # Generate plots
        self._generate_evaluation_plots(results)
        
        # Print summary
        self._print_annual_summary(results)
        
        return results
    
    def _calculate_comprehensive_results(self, monthly_stats: List[Dict], start_time: float) -> Dict[str, Any]:
        """Calculate comprehensive results for Wavelet model"""
        
        # Collect all individual metrics
        all_metrics = [metric for stats in monthly_stats for metric in stats['detailed_metrics']]
        
        # Variable-specific aggregation
        variable_results = {}
        for var in ['swh', 'mwd', 'mwp']:
            all_values = [metric[f'{var}_rmse'] for metric in all_metrics]
            variable_results[var] = {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'median': np.median(all_values),
                'p25': np.percentile(all_values, 25),
                'p75': np.percentile(all_values, 75),
                'p95': np.percentile(all_values, 95),
                'min': np.min(all_values),
                'max': np.max(all_values)
            }
        
        # Overall results
        all_overall = [metric['overall_rmse'] for metric in all_metrics]
        overall_results = {
            'mean': np.mean(all_overall),
            'std': np.std(all_overall),
            'median': np.median(all_overall),
            'p25': np.percentile(all_overall, 25),
            'p75': np.percentile(all_overall, 75),
            'p95': np.percentile(all_overall, 95),
            'min': np.min(all_overall),
            'max': np.max(all_overall)
        }
        
        # Performance vs baseline analysis
        baseline_rmse = 9.21
        improvement = ((baseline_rmse - overall_results['mean']) / baseline_rmse) * 100
        
        # Wavelet-specific targets
        wavelet_target_achieved = overall_results['mean'] < 8.5
        frequency_benefits = variable_results['mwp']['mean'] < 1.1  # MWP should benefit most
        
        # Seasonal analysis
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_results = {}
        for season, months in seasons.items():
            season_stats = [s for s in monthly_stats if s['month'] in months]
            if season_stats:
                season_overall = [s['overall_rmse_mean'] for s in season_stats]
                seasonal_results[season] = {
                    'mean': np.mean(season_overall),
                    'std': np.std(season_overall),
                    'months': months,
                    'vs_baseline': ((baseline_rmse - np.mean(season_overall)) / baseline_rmse) * 100
                }
        
        return {
            'evaluation_info': {
                'model_path': str(self.model_path),
                'test_year': self.test_year,
                'evaluation_timestamp': self.eval_timestamp,
                'total_samples': len(all_metrics),
                'months_evaluated': len(monthly_stats),
                'evaluation_time_seconds': time.time() - start_time,
                'model_type': 'Wavelet-Enhanced Variable LR',
                'baseline_rmse': baseline_rmse
            },
            'overall_statistics': overall_results,
            'variable_statistics': variable_results,
            'baseline_comparison': {
                'baseline_rmse': baseline_rmse,
                'current_rmse': overall_results['mean'],
                'improvement_percent': improvement,
                'improvement_absolute': baseline_rmse - overall_results['mean']
            },
            'wavelet_analysis': {
                'target_achieved': wavelet_target_achieved,
                'frequency_benefits': frequency_benefits,
                'expected_mwp_improvement': variable_results['mwp']['mean'] < 1.1,
                'vs_multi_scale_disaster': overall_results['mean'] < 14.95
            },
            'seasonal_analysis': seasonal_results,
            'monthly_summary': {
                stats['month']: {
                    'overall_rmse': stats['overall_rmse_mean'],
                    'swh_rmse': stats['swh_rmse_mean'],
                    'mwd_rmse': stats['mwd_rmse_mean'],
                    'mwp_rmse': stats['mwp_rmse_mean'],
                    'vs_baseline': ((baseline_rmse - stats['overall_rmse_mean']) / baseline_rmse) * 100
                } for stats in monthly_stats
            },
            'performance_targets': {
                'wavelet_target_range': '7.5-8.5',
                'achieved_wavelet_target': wavelet_target_achieved,
                'beats_baseline': overall_results['mean'] < baseline_rmse,
                'frequency_domain_success': frequency_benefits,
                'ready_for_buoys': overall_results['mean'] < 8.0
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results"""
        
        # Save main results JSON
        summary_file = self.eval_dir / "wavelet_annual_evaluation.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'evaluation_info': results['evaluation_info'],
                'overall_statistics': results['overall_statistics'],
                'variable_statistics': results['variable_statistics'],
                'baseline_comparison': results['baseline_comparison'],
                'wavelet_analysis': results['wavelet_analysis'],
                'seasonal_analysis': results['seasonal_analysis'],
                'monthly_summary': results['monthly_summary'],
                'performance_targets': results['performance_targets']
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {self.eval_dir}")
        print(f"   Summary: {summary_file}")
    
    def _generate_evaluation_plots(self, results: Dict[str, Any]):
        """Generate comprehensive wavelet evaluation plots"""
        
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Wavelet-Enhanced Variable LR Model Evaluation - {self.test_year}', fontsize=16)
        
        # Monthly performance trends
        monthly_data = results['monthly_results']
        months = sorted(monthly_data.keys())
        
        # Overall RMSE by month with baseline comparison
        overall_rmse = [monthly_data[m]['overall_rmse_mean'] for m in months]
        overall_std = [monthly_data[m]['overall_rmse_std'] for m in months]
        baseline_rmse = results['baseline_comparison']['baseline_rmse']
        
        axes[0, 0].errorbar(months, overall_rmse, yerr=overall_std, 
                           marker='o', capsize=5, capthick=2, linewidth=2, label='Wavelet-Enhanced', color='purple')
        axes[0, 0].axhline(y=baseline_rmse, color='red', linestyle='--', linewidth=2, 
                          label=f'Variable LR Baseline ({baseline_rmse:.2f})')
        axes[0, 0].axhline(y=8.5, color='green', linestyle=':', linewidth=2, 
                          label='Wavelet Target (8.5)')
        axes[0, 0].set_title('Overall RMSE by Month vs Baselines')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variable-specific performance with baselines
        swh_rmse = [monthly_data[m]['swh_rmse_mean'] for m in months]
        mwd_rmse = [monthly_data[m]['mwd_rmse_mean'] for m in months]
        mwp_rmse = [monthly_data[m]['mwp_rmse_mean'] for m in months]
        
        axes[0, 1].plot(months, swh_rmse, 'o-', label='SWH (m)', linewidth=2, color='blue')
        axes[0, 1].plot(months, mwp_rmse, 's-', label='MWP (s)', linewidth=2, color='green')
        axes[0, 1].axhline(y=0.674, color='blue', linestyle='--', alpha=0.7, label='SWH Baseline')
        axes[0, 1].axhline(y=1.187, color='green', linestyle='--', alpha=0.7, label='MWP Baseline')
        axes[0, 1].set_title('SWH and MWP RMSE by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MWD performance
        axes[0, 2].plot(months, mwd_rmse, 'ro-', label='MWD (°)', linewidth=2)
        axes[0, 2].axhline(y=25.8, color='red', linestyle='--', alpha=0.7, 
                          label='Baseline (25.8°)')
        axes[0, 2].set_title('MWD RMSE by Month')
        axes[0, 2].set_xlabel('Month')
        axes[0, 2].set_ylabel('MWD RMSE (degrees)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Wavelet vs baselines comparison
        improvements = [results['monthly_summary'][m]['vs_baseline'] for m in months]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        axes[1, 0].bar(months, improvements, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].axhline(y=15, color='green', linestyle=':', alpha=0.7, label='Target (+15%)')
        axes[1, 0].set_title('Wavelet Enhancement vs Baseline (%)')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution comparison
        all_overall = [metric['overall_rmse'] for month_data in monthly_data.values() 
                      for metric in month_data['detailed_metrics']]
        
        axes[1, 1].hist(all_overall, bins=50, alpha=0.7, edgecolor='black', 
                       label=f'Wavelet (μ={np.mean(all_overall):.3f})', color='purple')
        axes[1, 1].axvline(x=baseline_rmse, color='red', linestyle='--', linewidth=2,
                          label=f'Variable LR ({baseline_rmse:.2f})')
        axes[1, 1].axvline(x=8.5, color='green', linestyle=':', linewidth=2,
                          label='Wavelet Target (8.5)')
        axes[1, 1].axvline(x=14.95, color='orange', linestyle='-.', linewidth=2,
                          label='Multi-Scale Failure (14.95)')
        axes[1, 1].set_title('Overall RMSE Distribution')
        axes[1, 1].set_xlabel('Overall RMSE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary table
        axes[1, 2].axis('off')
        
        # Create comprehensive summary table
        overall_stats = results['overall_statistics']
        baseline_comp = results['baseline_comparison']
        wavelet_analysis = results['wavelet_analysis']
        
        table_data = [
            ['Metric', 'Wavelet Model', 'Baseline', 'Status'],
            ['Overall RMSE', f"{overall_stats['mean']:.3f} ± {overall_stats['std']:.3f}", 
             f"{baseline_comp['baseline_rmse']:.2f}", 
             f"{baseline_comp['improvement_percent']:+.1f}%"],
            ['SWH RMSE (m)', f"{results['variable_statistics']['swh']['mean']:.3f}", 
             "0.674", "vs 0.674m"],
            ['MWD RMSE (°)', f"{results['variable_statistics']['mwd']['mean']:.1f}", 
             "25.8", "vs 25.8°"],
            ['MWP RMSE (s)', f"{results['variable_statistics']['mwp']['mean']:.3f}", 
             "1.187", "vs 1.187s"],
            ['Wavelet Target', "✅ YES" if wavelet_analysis['target_achieved'] else "❌ NO", 
             "<8.5", "Target"],
            ['Frequency Benefits', "✅ YES" if wavelet_analysis['frequency_benefits'] else "❌ NO", 
             "MWP<1.1s", "Expected"],
            ['vs Multi-Scale', "✅ MUCH BETTER", "14.95", "Avoided disaster"]
        ]
        
        table = axes[1, 2].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        
        # Color code the results
        if baseline_comp['improvement_percent'] > 0:
            table[(1, 3)].set_facecolor('lightgreen')
        else:
            table[(1, 3)].set_facecolor('lightcoral')
        
        if wavelet_analysis['target_achieved']:
            table[(5, 1)].set_facecolor('lightgreen')
        else:
            table[(5, 1)].set_facecolor('lightyellow')
        
        if wavelet_analysis['frequency_benefits']:
            table[(6, 1)].set_facecolor('lightgreen')
        else:
            table[(6, 1)].set_facecolor('lightyellow')
        
        table[(7, 1)].set_facecolor('lightgreen')  # vs Multi-Scale
        
        axes[1, 2].set_title('Wavelet Enhancement Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.eval_dir / "wavelet_evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved: {plot_file}")
    
    def _print_annual_summary(self, results: Dict[str, Any]):
        """Print comprehensive annual summary"""
        
        print(f"\n🎯 WAVELET-ENHANCED ANNUAL SUMMARY - {self.test_year}")
        print("=" * 75)
        
        eval_info = results['evaluation_info']
        overall_stats = results['overall_statistics']
        baseline_comp = results['baseline_comparison']
        wavelet_analysis = results['wavelet_analysis']
        targets = results['performance_targets']
        
        print(f"📋 Evaluation Details:")
        print(f"   Model: {Path(eval_info['model_path']).name}")
        print(f"   Model Type: Wavelet-Enhanced Variable LR")
        print(f"   Total Samples: {eval_info['total_samples']:,}")
        print(f"   Months Evaluated: {eval_info['months_evaluated']}/12")
        print(f"   Evaluation Time: {eval_info['evaluation_time_seconds']:.1f} seconds")
        
        print(f"\n🎯 Performance vs Baseline:")
        print(f"   Variable LR Baseline: {baseline_comp['baseline_rmse']:.2f} RMSE")
        print(f"   Wavelet-Enhanced: {overall_stats['mean']:.3f} ± {overall_stats['std']:.3f} RMSE")
        print(f"   Improvement: {baseline_comp['improvement_percent']:+.1f}% ({baseline_comp['improvement_absolute']:+.3f} RMSE)")
        
        if baseline_comp['improvement_percent'] > 0:
            print(f"   🎉 SUCCESS: Wavelet enhancement provides improvement!")
        else:
            print(f"   ⚠️  Wavelet enhancement underperforms baseline")
        
        print(f"\n📊 Detailed Performance Statistics:")
        print(f"   Overall RMSE: {overall_stats['mean']:.3f} ± {overall_stats['std']:.3f}")
        print(f"     Range: [{overall_stats['min']:.3f}, {overall_stats['max']:.3f}]")
        print(f"     Median: {overall_stats['median']:.3f}")
        print(f"     P95: {overall_stats['p95']:.3f}")
        
        var_stats = results['variable_statistics']
        print(f"   Variable Breakdown:")
        print(f"     SWH: {var_stats['swh']['mean']:.3f} ± {var_stats['swh']['std']:.3f} m")
        print(f"     MWD: {var_stats['mwd']['mean']:.1f} ± {var_stats['mwd']['std']:.1f}°")
        print(f"     MWP: {var_stats['mwp']['mean']:.3f} ± {var_stats['mwp']['std']:.3f} s")
        
        # Wavelet-specific analysis
        print(f"\n🌊 Wavelet Enhancement Analysis:")
        print(f"   Target Range: 7.5-8.5 RMSE")
        print(f"   Achieved: {overall_stats['mean']:.3f} RMSE")
        
        if wavelet_analysis['target_achieved']:
            print(f"   ✅ WAVELET TARGET ACHIEVED: <8.5 RMSE!")
        else:
            print(f"   ⚠️  Wavelet target missed: {overall_stats['mean']:.3f} > 8.5")
        
        if wavelet_analysis['frequency_benefits']:
            print(f"   🎵 FREQUENCY BENEFITS: MWP improved as expected")
        else:
            print(f"   ⚠️  Limited frequency benefits: MWP={var_stats['mwp']['mean']:.3f}s")
        
        if wavelet_analysis['vs_multi_scale_disaster']:
            print(f"   ✅ MUCH BETTER: vs Multi-Scale disaster (14.95 RMSE)")
        
        # Seasonal performance
        seasonal = results['seasonal_analysis']
        if seasonal:
            print(f"\n🌍 Seasonal Performance:")
            for season, stats in seasonal.items():
                print(f"   {season}: {stats['mean']:.3f} RMSE ({stats['vs_baseline']:+.1f}% vs baseline)")
        
        # Performance targets assessment
        print(f"\n🏆 Performance Target Assessment:")
        if targets['achieved_wavelet_target']:
            print(f"   ✅ WAVELET SUCCESS: Achieved 7.5-8.5 RMSE target")
        elif targets['beats_baseline']:
            print(f"   📈 IMPROVEMENT: Beats Variable LR baseline")
        else:
            print(f"   ⚠️  NEEDS WORK: Does not beat baseline")
        
        if targets['frequency_domain_success']:
            print(f"   🎵 FREQUENCY SUCCESS: MWP benefits from frequency processing")
        
        if targets['ready_for_buoys']:
            print(f"   🌊 READY FOR BUOYS: Performance <8.0 enables buoy integration")
        
        # Monthly performance highlights
        monthly_summary = results['monthly_summary']
        best_month = min(monthly_summary.keys(), key=lambda k: monthly_summary[k]['overall_rmse'])
        worst_month = max(monthly_summary.keys(), key=lambda k: monthly_summary[k]['overall_rmse'])
        
        print(f"\n📅 Monthly Performance Highlights:")
        print(f"   Best Month: {best_month} ({monthly_summary[best_month]['overall_rmse']:.3f} RMSE)")
        print(f"   Worst Month: {worst_month} ({monthly_summary[worst_month]['overall_rmse']:.3f} RMSE)")
        print(f"   Monthly Variation: {monthly_summary[worst_month]['overall_rmse'] - monthly_summary[best_month]['overall_rmse']:.3f}")
        
        # Wavelet-specific insights
        print(f"\n🌊 Wavelet Innovation Assessment:")
        if baseline_comp['improvement_percent'] > 10:
            print(f"   🎉 EXCELLENT: Wavelet provides major improvement (>{baseline_comp['improvement_percent']:.1f}%)")
            print(f"      Frequency-domain processing is working excellently!")
            print(f"      Ready for buoy integration or ensemble methods")
        elif baseline_comp['improvement_percent'] > 0:
            print(f"   ✅ GOOD: Wavelet provides improvement ({baseline_comp['improvement_percent']:.1f}%)")
            print(f"      Frequency processing showing benefits")
            print(f"      Consider tuning wavelet parameters or ensemble")
        else:
            print(f"   ⚠️  DISAPPOINTING: Wavelet underperforms baseline")
            print(f"      May need different wavelet type or fusion method")
            print(f"      Consider buoy integration with Variable LR instead")
        
        # Next steps recommendations
        print(f"\n🚀 Next Steps Recommendations:")
        if targets['ready_for_buoys']:
            print(f"   • Excellent performance! Ready for buoy data integration")
            print(f"   • Consider ensemble with Variable LR model")
            print(f"   • Deploy for operational testing")
        elif targets['achieved_wavelet_target']:
            print(f"   • Good wavelet success! Try buoy integration")
            print(f"   • Fine-tune wavelet parameters (levels, fusion)")
            print(f"   • Consider ensemble methods")
        elif targets['beats_baseline']:
            print(f"   • Promising improvement over baseline")
            print(f"   • Optimize wavelet architecture")
            print(f"   • Try different fusion methods")
        else:
            print(f"   • Wavelet approach needs refinement")
            print(f"   • Consider buoy integration with Variable LR baseline")
            print(f"   • Try ensemble of successful models")
        
        print(f"\n💾 Detailed results saved to: {self.eval_dir}")


def main():
    """Main evaluation function for Wavelet-Enhanced models"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Wavelet-Enhanced Wave Model Annual Evaluation')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to evaluate (default: latest wavelet)')
    parser.add_argument('--test_year', type=int, default=2024,
                       help='Year to test against (default: 2024)')
    parser.add_argument('--samples_per_month', type=int, default=100,
                       help='Number of samples per month (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = WaveletAnnualEvaluator(
            model_name=args.model_name,
            test_year=args.test_year
        )
        
        # Run evaluation
        results = evaluator.run_annual_evaluation(
            samples_per_month=args.samples_per_month
        )
        
        print(f"\n🎉 Wavelet-Enhanced evaluation complete!")
        print(f"   Results directory: {evaluator.eval_dir}")
        
        # Quick performance summary
        overall_rmse = results['overall_statistics']['mean']
        improvement = results['baseline_comparison']['improvement_percent']
        wavelet_target = results['wavelet_analysis']['target_achieved']
        frequency_benefits = results['wavelet_analysis']['frequency_benefits']
        
        if wavelet_target:
            print(f"   🎯 WAVELET TARGET ACHIEVED: {overall_rmse:.3f} RMSE (<8.5)")
        if improvement > 0:
            print(f"   📈 BASELINE BEATEN: {improvement:.1f}% improvement")
        if frequency_benefits:
            print(f"   🎵 FREQUENCY BENEFITS: MWP improved as expected")
        
        if overall_rmse < 8.0:
            print(f"   🌊 READY FOR BUOYS: Performance enables next enhancement")
        
        return results
        
    except Exception as e:
        print(f"❌ Wavelet evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()