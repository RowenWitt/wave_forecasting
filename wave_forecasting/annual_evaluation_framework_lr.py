#!/usr/bin/env python3
"""
Annual Evaluation Framework for Wave Models
Comprehensive testing against full year with detailed statistics
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
from circular_mwd_evaluator import CircularMWDEvaluator
from mwd_circular_fixes import evaluate_model_with_circular_metrics, compute_circular_rmse
from spatiotemporal_with_circular_mwd import SpatioTemporalConfig
from variable_lr_spatiotemporal_with_circular_mwd import VariableLRConfig, VariableLRSpatioTemporalGNN

class AnnualEvaluator:
    """Comprehensive annual evaluation framework"""
    
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
        self.model_path = self._find_model_to_evaluate()
        if not self.model_path:
            raise ValueError("No suitable model found for evaluation")
        
        # Setup evaluation directory
        self.eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.model_path.parent / f"evaluation_{self.eval_timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 Annual Wave Model Evaluation Framework")
        print(f"   Model: {self.model_path}")
        print(f"   Test Year: {self.test_year}")
        print(f"   Results: {self.eval_dir}")
        print(f"   Device: {self.device}")
        
        # Results storage
        self.results = defaultdict(list)
        self.monthly_results = {}
        self.spatial_results = {}
        
    def _find_model_to_evaluate(self) -> Path:
        """Find the model to evaluate based on criteria"""
        
        experiments_dir = Path("experiments")
        
        if self.model_name:
            # Look for specific model name
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.model_name.lower() in d.name.lower()]
        else:
            # Find latest circular model by default
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and "circular" in d.name.lower()]
        
        if not model_experiments:
            print(f"❌ No models found matching criteria")
            return None
        
        # Get most recent experiment
        latest_experiment = max(model_experiments, key=lambda x: x.stat().st_mtime)
        
        # Look for model file
        model_candidates = [
            latest_experiment / "spatiotemporal_circular_model.pt",
            latest_experiment / "best_model.pt",
            latest_experiment / "best_variable_lr_model.pt",
            latest_experiment / "final_model.pt"
        ]

        for model_path in model_candidates:
            if model_path.exists():
                print(f"✅ Found model: {model_path}")
                return model_path
        
        print(model_candidates)

        print(f"❌ No model file found in {latest_experiment}")
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
        prediction_horizon = 4
        
        max_time = len(era5_atmo.valid_time) - sequence_length - prediction_horizon
        step_size = max(1, max_time // samples_per_month)  # Ensure good coverage
        
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
                
                # Target (single step for now)
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
    
    def evaluate_month(self, evaluator: CircularMWDEvaluator, 
                      era5_atmo, era5_waves, gebco_data, month: int) -> Dict[str, Any]:
        """Evaluate model performance for a specific month"""
        
        print(f"🔍 Evaluating {self.test_year}-{month:02d}...")
        
        # Create test sequences
        test_sequences = self.create_test_sequences(era5_atmo, era5_waves, gebco_data, month)
        
        if not test_sequences:
            print(f"   ⚠️  No test sequences for month {month}")
            return {}
        
        # Evaluate all sequences
        month_metrics = []
        predictions_list = []
        targets_list = []
        
        for i, sequence in enumerate(test_sequences):
            if i % 20 == 0:
                print(f"     Sample {i+1}/{len(test_sequences)}")
            
            try:
                # Make prediction
                predictions = evaluator.predict(sequence['input'], multi_step=False)
                targets = sequence['target']
                
                # Convert to numpy
                pred_np = predictions.cpu().numpy()
                target_np = targets.cpu().numpy()
                
                # Calculate metrics
                metrics = evaluate_model_with_circular_metrics(pred_np, target_np)
                metrics['month'] = month
                metrics['timestep'] = sequence['timestep']
                metrics['datetime'] = sequence['datetime']
                
                month_metrics.append(metrics)
                predictions_list.append(pred_np)
                targets_list.append(target_np)
                
            except Exception as e:
                print(f"       ⚠️  Failed on sample {i}: {e}")
                continue
        
        if not month_metrics:
            return {}
        
        # Aggregate monthly statistics
        monthly_stats = {
            'month': month,
            'num_samples': len(month_metrics),
            'swh_rmse_mean': np.mean([m['swh_rmse'] for m in month_metrics]),
            'swh_rmse_std': np.std([m['swh_rmse'] for m in month_metrics]),
            'mwd_rmse_mean': np.mean([m['mwd_rmse'] for m in month_metrics]),
            'mwd_rmse_std': np.std([m['mwd_rmse'] for m in month_metrics]),
            'mwp_rmse_mean': np.mean([m['mwp_rmse'] for m in month_metrics]),
            'mwp_rmse_std': np.std([m['mwp_rmse'] for m in month_metrics]),
            'overall_rmse_mean': np.mean([m['overall_rmse'] for m in month_metrics]),
            'overall_rmse_std': np.std([m['overall_rmse'] for m in month_metrics])
        }
        
        # Store detailed results
        monthly_stats['detailed_metrics'] = month_metrics
        monthly_stats['predictions'] = predictions_list
        monthly_stats['targets'] = targets_list
        
        print(f"   ✅ Month {month}: Overall RMSE = {monthly_stats['overall_rmse_mean']:.3f} ± {monthly_stats['overall_rmse_std']:.3f}")
        
        return monthly_stats
    
    def run_annual_evaluation(self, samples_per_month: int = 100) -> Dict[str, Any]:
        """Run comprehensive annual evaluation"""
        
        print(f"🌊 ANNUAL WAVE MODEL EVALUATION - {self.test_year}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load model
        evaluator = CircularMWDEvaluator(str(self.model_path))
        print(f"✅ Model loaded successfully")
        
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
        
        # Aggregate annual statistics
        annual_stats = self._calculate_annual_statistics(all_monthly_stats)
        
        # Generate detailed results
        results = {
            'evaluation_info': {
                'model_path': str(self.model_path),
                'test_year': self.test_year,
                'evaluation_timestamp': self.eval_timestamp,
                'samples_per_month': samples_per_month,
                'total_samples': sum(stats['num_samples'] for stats in all_monthly_stats),
                'months_evaluated': len(all_monthly_stats),
                'evaluation_time_seconds': time.time() - start_time
            },
            'annual_statistics': annual_stats,
            'monthly_results': monthly_results,
            'spatial_analysis': self._calculate_spatial_statistics(all_monthly_stats),
            'temporal_analysis': self._calculate_temporal_statistics(all_monthly_stats)
        }
        
        # Save results
        self._save_results(results)
        
        # Generate plots
        self._generate_evaluation_plots(results)
        
        # Print summary
        self._print_annual_summary(results)
        
        return results
    
    def _calculate_annual_statistics(self, monthly_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive annual statistics"""
        
        # Collect all individual metrics
        all_swh = [metric['swh_rmse'] for stats in monthly_stats 
                  for metric in stats['detailed_metrics']]
        all_mwd = [metric['mwd_rmse'] for stats in monthly_stats 
                  for metric in stats['detailed_metrics']]
        all_mwp = [metric['mwp_rmse'] for stats in monthly_stats 
                  for metric in stats['detailed_metrics']]
        all_overall = [metric['overall_rmse'] for stats in monthly_stats 
                      for metric in stats['detailed_metrics']]
        
        return {
            'swh_rmse': {
                'mean': np.mean(all_swh),
                'std': np.std(all_swh),
                'median': np.median(all_swh),
                'p25': np.percentile(all_swh, 25),
                'p75': np.percentile(all_swh, 75),
                'p95': np.percentile(all_swh, 95),
                'min': np.min(all_swh),
                'max': np.max(all_swh)
            },
            'mwd_rmse': {
                'mean': np.mean(all_mwd),
                'std': np.std(all_mwd),
                'median': np.median(all_mwd),
                'p25': np.percentile(all_mwd, 25),
                'p75': np.percentile(all_mwd, 75),
                'p95': np.percentile(all_mwd, 95),
                'min': np.min(all_mwd),
                'max': np.max(all_mwd)
            },
            'mwp_rmse': {
                'mean': np.mean(all_mwp),
                'std': np.std(all_mwp),
                'median': np.median(all_mwp),
                'p25': np.percentile(all_mwp, 25),
                'p75': np.percentile(all_mwp, 75),
                'p95': np.percentile(all_mwp, 95),
                'min': np.min(all_mwp),
                'max': np.max(all_mwp)
            },
            'overall_rmse': {
                'mean': np.mean(all_overall),
                'std': np.std(all_overall),
                'median': np.median(all_overall),
                'p25': np.percentile(all_overall, 25),
                'p75': np.percentile(all_overall, 75),
                'p95': np.percentile(all_overall, 95),
                'min': np.min(all_overall),
                'max': np.max(all_overall)
            },
            'sample_counts': {
                'total_samples': len(all_overall),
                'monthly_samples': [stats['num_samples'] for stats in monthly_stats]
            }
        }
    
    def _calculate_spatial_statistics(self, monthly_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate spatial distribution of errors"""
        
        # For now, aggregate by month (could extend to lat/lon analysis)
        monthly_means = {}
        for stats in monthly_stats:
            month = stats['month']
            monthly_means[month] = {
                'swh_rmse': stats['swh_rmse_mean'],
                'mwd_rmse': stats['mwd_rmse_mean'],
                'mwp_rmse': stats['mwp_rmse_mean'],
                'overall_rmse': stats['overall_rmse_mean']
            }
        
        return {
            'monthly_means': monthly_means,
            'best_month': min(monthly_means.keys(), 
                            key=lambda k: monthly_means[k]['overall_rmse']),
            'worst_month': max(monthly_means.keys(), 
                             key=lambda k: monthly_means[k]['overall_rmse'])
        }
    
    def _calculate_temporal_statistics(self, monthly_stats: List[Dict]) -> Dict[str, Any]:
        """Calculate temporal trends and patterns"""
        
        monthly_performance = {}
        for stats in monthly_stats:
            month = stats['month']
            monthly_performance[month] = stats['overall_rmse_mean']
        
        # Seasonal analysis
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5], 
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_performance = {}
        for season, months in seasons.items():
            season_errors = [monthly_performance[m] for m in months 
                           if m in monthly_performance]
            if season_errors:
                seasonal_performance[season] = {
                    'mean': np.mean(season_errors),
                    'std': np.std(season_errors),
                    'months': months
                }
        
        return {
            'monthly_performance': monthly_performance,
            'seasonal_performance': seasonal_performance,
            'temporal_trend': self._calculate_temporal_trend(monthly_stats)
        }
    
    def _calculate_temporal_trend(self, monthly_stats: List[Dict]) -> Dict[str, float]:
        """Calculate if there's a temporal trend in performance"""
        
        months = [stats['month'] for stats in monthly_stats]
        performance = [stats['overall_rmse_mean'] for stats in monthly_stats]
        
        if len(months) < 3:
            return {'trend_slope': 0.0, 'correlation': 0.0}
        
        # Simple linear trend
        correlation = np.corrcoef(months, performance)[0, 1]
        slope = np.polyfit(months, performance, 1)[0]
        
        return {
            'trend_slope': slope,
            'correlation': correlation,
            'interpretation': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results"""
        
        # Save main results JSON (without detailed data to keep size manageable)
        summary_results = {
            'evaluation_info': results['evaluation_info'],
            'annual_statistics': results['annual_statistics'],
            'spatial_analysis': results['spatial_analysis'],
            'temporal_analysis': results['temporal_analysis'],
            'monthly_summary': {
                month: {
                    'month': data['month'],
                    'num_samples': data['num_samples'],
                    'swh_rmse_mean': data['swh_rmse_mean'],
                    'mwd_rmse_mean': data['mwd_rmse_mean'],
                    'mwp_rmse_mean': data['mwp_rmse_mean'],
                    'overall_rmse_mean': data['overall_rmse_mean']
                }
                for month, data in results['monthly_results'].items()
            }
        }
        
        # Save summary
        summary_file = self.eval_dir / "annual_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2, default=str)
        
        # Save detailed monthly data
        for month, data in results['monthly_results'].items():
            month_file = self.eval_dir / f"month_{month:02d}_detailed.json"
            with open(month_file, 'w') as f:
                json.dump({
                    'month': month,
                    'statistics': {k: v for k, v in data.items() 
                                 if k not in ['detailed_metrics', 'predictions', 'targets']},
                    'metrics': data.get('detailed_metrics', [])
                }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {self.eval_dir}")
        print(f"   Summary: {summary_file}")
        print(f"   Monthly details: {len(results['monthly_results'])} files")
    
    def _generate_evaluation_plots(self, results: Dict[str, Any]):
        """Generate comprehensive evaluation plots"""
        
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Annual Wave Model Evaluation - {self.test_year}', fontsize=16)
        
        # Monthly performance trends
        monthly_data = results['monthly_results']
        months = sorted(monthly_data.keys())
        
        # Overall RMSE by month
        overall_rmse = [monthly_data[m]['overall_rmse_mean'] for m in months]
        overall_std = [monthly_data[m]['overall_rmse_std'] for m in months]
        
        axes[0, 0].errorbar(months, overall_rmse, yerr=overall_std, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
        axes[0, 0].set_title('Overall RMSE by Month')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Variable-specific performance
        swh_rmse = [monthly_data[m]['swh_rmse_mean'] for m in months]
        mwd_rmse = [monthly_data[m]['mwd_rmse_mean'] for m in months]
        mwp_rmse = [monthly_data[m]['mwp_rmse_mean'] for m in months]
        
        axes[0, 1].plot(months, swh_rmse, 'o-', label='SWH (m)', linewidth=2)
        axes[0, 1].plot(months, mwp_rmse, 's-', label='MWP (s)', linewidth=2)
        axes[0, 1].set_title('SWH and MWP RMSE by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MWD on separate scale
        ax_mwd = axes[0, 2]
        ax_mwd.plot(months, mwd_rmse, 'ro-', label='MWD (°)', linewidth=2)
        ax_mwd.set_title('MWD RMSE by Month')
        ax_mwd.set_xlabel('Month')
        ax_mwd.set_ylabel('MWD RMSE (degrees)')
        ax_mwd.grid(True, alpha=0.3)
        
        # Error distribution histograms
        all_overall = [metric['overall_rmse'] for month_data in monthly_data.values() 
                      for metric in month_data['detailed_metrics']]
        all_mwd = [metric['mwd_rmse'] for month_data in monthly_data.values() 
                  for metric in month_data['detailed_metrics']]
        
        axes[1, 0].hist(all_overall, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Overall RMSE Distribution')
        axes[1, 0].set_xlabel('Overall RMSE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(all_mwd, bins=50, alpha=0.7, edgecolor='black', color='red')
        axes[1, 1].set_title('MWD RMSE Distribution')
        axes[1, 1].set_xlabel('MWD RMSE (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary table
        axes[1, 2].axis('off')
        annual_stats = results['annual_statistics']
        
        table_data = [
            ['Metric', 'Mean', 'Median', 'Std', 'P95'],
            ['Overall RMSE', f"{annual_stats['overall_rmse']['mean']:.3f}", 
             f"{annual_stats['overall_rmse']['median']:.3f}",
             f"{annual_stats['overall_rmse']['std']:.3f}",
             f"{annual_stats['overall_rmse']['p95']:.3f}"],
            ['SWH RMSE (m)', f"{annual_stats['swh_rmse']['mean']:.3f}", 
             f"{annual_stats['swh_rmse']['median']:.3f}",
             f"{annual_stats['swh_rmse']['std']:.3f}",
             f"{annual_stats['swh_rmse']['p95']:.3f}"],
            ['MWD RMSE (°)', f"{annual_stats['mwd_rmse']['mean']:.1f}", 
             f"{annual_stats['mwd_rmse']['median']:.1f}",
             f"{annual_stats['mwd_rmse']['std']:.1f}",
             f"{annual_stats['mwd_rmse']['p95']:.1f}"],
            ['MWP RMSE (s)', f"{annual_stats['mwp_rmse']['mean']:.3f}", 
             f"{annual_stats['mwp_rmse']['median']:.3f}",
             f"{annual_stats['mwp_rmse']['std']:.3f}",
             f"{annual_stats['mwp_rmse']['p95']:.3f}"]
        ]
        
        table = axes[1, 2].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 2].set_title('Annual Statistics Summary')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.eval_dir / "annual_evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved: {plot_file}")
    
    def _print_annual_summary(self, results: Dict[str, Any]):
        """Print comprehensive annual summary"""
        
        print(f"\n🎯 ANNUAL EVALUATION SUMMARY - {self.test_year}")
        print("=" * 60)
        
        eval_info = results['evaluation_info']
        annual_stats = results['annual_statistics']
        
        print(f"📋 Evaluation Details:")
        print(f"   Model: {Path(eval_info['model_path']).name}")
        print(f"   Total Samples: {eval_info['total_samples']:,}")
        print(f"   Months Evaluated: {eval_info['months_evaluated']}/12")
        print(f"   Evaluation Time: {eval_info['evaluation_time_seconds']:.1f} seconds")
        
        print(f"\n📊 Performance Statistics:")
        print(f"   Overall RMSE: {annual_stats['overall_rmse']['mean']:.3f} ± {annual_stats['overall_rmse']['std']:.3f}")
        print(f"   SWH RMSE: {annual_stats['swh_rmse']['mean']:.3f} ± {annual_stats['swh_rmse']['std']:.3f} m")
        print(f"   MWD RMSE: {annual_stats['mwd_rmse']['mean']:.1f} ± {annual_stats['mwd_rmse']['std']:.1f}°")
        print(f"   MWP RMSE: {annual_stats['mwp_rmse']['mean']:.3f} ± {annual_stats['mwp_rmse']['std']:.3f} s")
        
        print(f"\n📈 Performance Ranges:")
        print(f"   Overall RMSE: [{annual_stats['overall_rmse']['min']:.3f}, {annual_stats['overall_rmse']['max']:.3f}]")
        print(f"   MWD RMSE: [{annual_stats['mwd_rmse']['min']:.1f}°, {annual_stats['mwd_rmse']['max']:.1f}°]")
        
        # Seasonal performance
        seasonal = results['temporal_analysis']['seasonal_performance']
        if seasonal:
            print(f"\n🌍 Seasonal Performance (Overall RMSE):")
            for season, stats in seasonal.items():
                print(f"   {season}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        # Best/worst months
        spatial = results['spatial_analysis']
        best_month = spatial['best_month']
        worst_month = spatial['worst_month']
        best_rmse = spatial['monthly_means'][best_month]['overall_rmse']
        worst_rmse = spatial['monthly_means'][worst_month]['overall_rmse']
        
        print(f"\n🏆 Monthly Performance:")
        print(f"   Best Month: {best_month} (RMSE: {best_rmse:.3f})")
        print(f"   Worst Month: {worst_month} (RMSE: {worst_rmse:.3f})")
        print(f"   Seasonal Variation: {worst_rmse - best_rmse:.3f}")
        
        # Performance assessment
        print(f"\n🎯 Performance Assessment:")
        mean_overall = annual_stats['overall_rmse']['mean']
        mean_mwd = annual_stats['mwd_rmse']['mean']
        
        if mean_overall < 5.0:
            print(f"   🎉 EXCELLENT: Overall RMSE < 5.0 (Publication quality)")
        elif mean_overall < 10.0:
            print(f"   ✅ GOOD: Overall RMSE < 10.0 (Strong performance)")
        elif mean_overall < 20.0:
            print(f"   📈 PROMISING: Overall RMSE < 20.0 (Good baseline)")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: Overall RMSE > 20.0")
        
        if mean_mwd < 30.0:
            print(f"   🎯 EXCELLENT MWD: < 30° (Competitive with operational models)")
        elif mean_mwd < 50.0:
            print(f"   ✅ GOOD MWD: < 50° (Strong directional accuracy)")
        elif mean_mwd < 80.0:
            print(f"   📈 MODERATE MWD: < 80° (Room for improvement)")
        else:
            print(f"   ⚠️  POOR MWD: > 80° (Needs attention)")
        
        print(f"\n💾 Detailed results saved to: {self.eval_dir}")


def main():
    """Main evaluation function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Annual Wave Model Evaluation')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to evaluate (default: latest circular)')
    parser.add_argument('--test_year', type=int, default=2024,
                       help='Year to test against (default: 2024)')
    parser.add_argument('--samples_per_month', type=int, default=100,
                       help='Number of samples per month (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = VariableLRAnnualEvaluator(
            model_name=args.model_name,
            test_year=args.test_year
        )
        
        # Run evaluation
        results = evaluator.run_annual_evaluation(
            samples_per_month=args.samples_per_month
        )
        
        print(f"\n🎉 Annual evaluation complete!")
        print(f"   Results directory: {evaluator.eval_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None




from variable_lr_spatiotemporal_with_circular_mwd import VariableLRConfig, VariableLRSpatioTemporalGNN, VariableSpecificLoss

class VariableLREvaluator:
    """Dedicated evaluator for Variable Learning Rate models"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"🔧 Variable LR Evaluator")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        
        # Model components
        self.model = None
        self.config = None
        self.normalizer = None
        self.target_normalizer = None
        self.edge_index = None
        self.edge_attr = None
        
        # Results
        self.results = {}
        
    def load_model(self):
        """Load the variable LR model"""
        print(f"📦 Loading Variable LR model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract components
            self.config = checkpoint.get('config', VariableLRConfig())
            self.normalizer = checkpoint.get('feature_normalizer')
            self.target_normalizer = checkpoint.get('target_normalizer')
            self.edge_index = checkpoint.get('edge_index')
            self.edge_attr = checkpoint.get('edge_attr')
            
            # Create model
            self.model = VariableLRSpatioTemporalGNN(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"  ✅ Model loaded successfully")
            print(f"     Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"     Hidden dim: {self.config.hidden_dim}")
            print(f"     Variable-specific heads: SWH, MWD (circular), MWP")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
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
        """Make prediction with variable LR model"""
        
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
        """Evaluate model on single test sequence"""
        
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
            return {
                'swh_rmse': float('inf'),
                'mwd_rmse': float('inf'),
                'mwp_rmse': float('inf'),
                'overall_rmse': float('inf')
            }

class VariableLRAnnualEvaluator:
    """Annual evaluation framework for Variable LR models"""
    
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
        self.model_path = self._find_model_to_evaluate()
        if not self.model_path:
            raise ValueError("No suitable Variable LR model found for evaluation")
        
        # Setup evaluation directory
        self.eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.model_path.parent / f"evaluation_{self.eval_timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 Variable LR Annual Evaluation Framework")
        print(f"   Model: {self.model_path}")
        print(f"   Test Year: {self.test_year}")
        print(f"   Results: {self.eval_dir}")
        print(f"   Device: {self.device}")
        
        # Results storage
        self.results = defaultdict(list)
        self.monthly_results = {}
        
    def _find_model_to_evaluate(self) -> Path:
        """Find the Variable LR model to evaluate"""
        
        experiments_dir = Path("experiments")
        
        if self.model_name:
            # Look for specific model name
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.model_name.lower() in d.name.lower()]
        else:
            # Find latest variable LR model by default
            model_experiments = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and "variable_lr" in d.name.lower()]
        
        if not model_experiments:
            print(f"❌ No Variable LR models found matching criteria")
            return None
        
        # Get most recent experiment
        latest_experiment = max(model_experiments, key=lambda x: x.stat().st_mtime)
        
        # Look for model file
        model_candidates = [
            latest_experiment / "variable_lr_circular_model.pt",
            latest_experiment / "best_variable_lr_model.pt",
            latest_experiment / "final_model.pt"
        ]
        
        for model_path in model_candidates:
            if model_path.exists():
                print(f"✅ Found Variable LR model: {model_path}")
                return model_path
        
        print(f"❌ No model file found in {latest_experiment}")
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
        prediction_horizon = 1  # Single step for Variable LR model
        
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
    
    def evaluate_month(self, evaluator: VariableLREvaluator, 
                      era5_atmo, era5_waves, gebco_data, month: int) -> Dict[str, Any]:
        """Evaluate Variable LR model performance for a specific month"""
        
        print(f"🔍 Evaluating Variable LR model on {self.test_year}-{month:02d}...")
        
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
        
        # Variable-specific performance analysis
        monthly_stats['variable_performance'] = {
            'swh': {
                'mean': monthly_stats['swh_rmse_mean'],
                'std': monthly_stats['swh_rmse_std'],
                'min': np.min(variable_errors['swh']),
                'max': np.max(variable_errors['swh']),
                'median': np.median(variable_errors['swh'])
            },
            'mwd': {
                'mean': monthly_stats['mwd_rmse_mean'],
                'std': monthly_stats['mwd_rmse_std'],
                'min': np.min(variable_errors['mwd']),
                'max': np.max(variable_errors['mwd']),
                'median': np.median(variable_errors['mwd'])
            },
            'mwp': {
                'mean': monthly_stats['mwp_rmse_mean'],
                'std': monthly_stats['mwp_rmse_std'],
                'min': np.min(variable_errors['mwp']),
                'max': np.max(variable_errors['mwp']),
                'median': np.median(variable_errors['mwp'])
            }
        }
        
        # Store detailed results
        monthly_stats['detailed_metrics'] = month_metrics
        
        print(f"   ✅ Month {month}: Overall RMSE = {monthly_stats['overall_rmse_mean']:.3f} ± {monthly_stats['overall_rmse_std']:.3f}")
        print(f"       Variable breakdown: SWH={monthly_stats['swh_rmse_mean']:.3f}, MWD={monthly_stats['mwd_rmse_mean']:.1f}°, MWP={monthly_stats['mwp_rmse_mean']:.3f}")
        
        return monthly_stats
    
    def run_annual_evaluation(self, samples_per_month: int = 100) -> Dict[str, Any]:
        """Run comprehensive annual evaluation for Variable LR model"""
        
        print(f"🌊 VARIABLE LR ANNUAL EVALUATION - {self.test_year}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load model
        evaluator = VariableLREvaluator(str(self.model_path))
        if not evaluator.load_model():
            raise ValueError("Failed to load Variable LR model")
        
        print(f"✅ Variable LR model loaded successfully")
        
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
        """Calculate comprehensive results for Variable LR model"""
        
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
                    'months': months
                }
        
        return {
            'evaluation_info': {
                'model_path': str(self.model_path),
                'test_year': self.test_year,
                'evaluation_timestamp': self.eval_timestamp,
                'total_samples': len(all_metrics),
                'months_evaluated': len(monthly_stats),
                'evaluation_time_seconds': time.time() - start_time,
                'model_type': 'Variable Learning Rate'
            },
            'overall_statistics': overall_results,
            'variable_statistics': variable_results,
            'seasonal_analysis': seasonal_results,
            'monthly_summary': {
                stats['month']: {
                    'overall_rmse': stats['overall_rmse_mean'],
                    'swh_rmse': stats['swh_rmse_mean'],
                    'mwd_rmse': stats['mwd_rmse_mean'],
                    'mwp_rmse': stats['mwp_rmse_mean']
                } for stats in monthly_stats
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive evaluation results"""
        
        # Save main results JSON
        summary_file = self.eval_dir / "variable_lr_annual_evaluation.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'evaluation_info': results['evaluation_info'],
                'overall_statistics': results['overall_statistics'],
                'variable_statistics': results['variable_statistics'],
                'seasonal_analysis': results['seasonal_analysis'],
                'monthly_summary': results['monthly_summary']
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {self.eval_dir}")
        print(f"   Summary: {summary_file}")
    
if __name__ == "__main__":
    main()