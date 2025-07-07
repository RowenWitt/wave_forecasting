# prediction/forecasting.py
"""
Clean autoregressive wave forecasting system
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import time
from datetime import datetime


@dataclass
class ForecastConfig:
    """Configuration for forecasting parameters"""
    max_horizon_hours: int = 168  # 7 days
    time_step_hours: int = 6
    initial_time_idx: int = 20
    
    @property
    def max_steps(self) -> int:
        return self.max_horizon_hours // self.time_step_hours


@dataclass
class ForecastResult:
    """Container for forecast results"""
    predictions: Dict[int, torch.Tensor]  # hour -> prediction tensor
    metadata: Dict[str, any]
    performance: Optional[Dict[int, Dict[str, float]]] = None


class WavePredictor:
    """Loads and manages the trained spatial wave model"""
    
    def __init__(self, model, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                 feature_names: List[str], experiment_id: str):
        """
        Args:
            model: Trained SpatialWaveGNN
            edge_index: Graph edge connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_features]
            feature_names: List of input feature names
            experiment_id: Identifier for the model
        """
        self.model = model
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.feature_names = feature_names
        self.experiment_id = experiment_id
        
        self.model.eval()
        
        # Find wave variable indices for state updates
        self.wave_indices = self._find_wave_indices()
        
        print(f"🔧 WavePredictor initialized:")
        print(f"   Model: {experiment_id}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Wave indices: {self.wave_indices}")
    
    def _find_wave_indices(self) -> List[int]:
        """Find indices of wave variables in feature vector"""
        wave_vars = ['swh', 'mwd', 'mwp']
        indices = []
        
        for wave_var in wave_vars:
            for i, feature_name in enumerate(self.feature_names):
                if wave_var in feature_name.lower():
                    indices.append(i)
                    break
        
        return indices
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Single timestep prediction
        
        Args:
            features: Input features [num_nodes, num_features]
            
        Returns:
            Wave predictions [num_nodes, 3] for [SWH, MWD, MWP]
        """
        with torch.no_grad():
            return self.model(features, self.edge_index, self.edge_attr)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, data_setup_func=None) -> 'WavePredictor':
        """
        Load predictor from model checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            data_setup_func: Optional function to setup data environment
            
        Returns:
            Configured WavePredictor
        """
        from models.spatial import SpatialWaveGNN
        from config.base import ModelConfig
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        config_dict = checkpoint.get('config', {})
        exp_id = checkpoint.get('experiment_id', checkpoint_path.stem)
        
        # Recreate model config
        if isinstance(config_dict, dict) and 'model' in config_dict:
            model_config_dict = config_dict['model']
            model_config = ModelConfig(
                hidden_dim=model_config_dict.get('hidden_dim', 256),
                num_spatial_layers=model_config_dict.get('num_spatial_layers', 12),
                edge_features=model_config_dict.get('edge_features', 3),
                output_features=model_config_dict.get('output_features', 3)
            )
        else:
            model_config = ModelConfig()
        
        # Add input features from state dict
        if 'encoder.0.weight' in state_dict:
            model_config.input_features = state_dict['encoder.0.weight'].shape[1]
        
        # Create and load model
        model = SpatialWaveGNN(model_config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Setup data environment if function provided
        if data_setup_func:
            mesh_loader, edge_index, edge_attr, feature_names = data_setup_func()
        else:
            # Default data setup
            edge_index, edge_attr, feature_names = cls._default_data_setup()
        
        return cls(model, edge_index, edge_attr, feature_names, exp_id)
    
    @staticmethod
    def _default_data_setup():
        """Default data environment setup"""
        from data.loaders import ERA5DataManager, GEBCODataManager
        from data.preprocessing import MultiResolutionInterpolator
        from data.datasets import MeshDataLoader
        from mesh.icosahedral import IcosahedralMesh
        from mesh.connectivity import compute_regional_edges
        from config.base import DataConfig, MeshConfig
        
        # Setup configs
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        # Load data
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
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
        
        # Get feature names
        sample_data = mesh_loader.load_features(time_idx=0)
        feature_names = sample_data['feature_names']
        
        # Create graph
        region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
        
        return edge_index, edge_attr, feature_names


class AutoregressiveForecaster:
    """Runs autoregressive forecasts using a WavePredictor"""
    
    def __init__(self, predictor: WavePredictor, data_loader):
        """
        Args:
            predictor: Configured WavePredictor
            data_loader: Data loader for getting initial conditions and ground truth
        """
        self.predictor = predictor
        self.data_loader = data_loader
    
    def forecast(self, config: ForecastConfig) -> ForecastResult:
        """
        Run autoregressive forecast
        
        Args:
            config: Forecast configuration
            
        Returns:
            ForecastResult containing predictions and metadata
        """
        print(f"🔮 Running autoregressive forecast:")
        print(f"   Initial time: {config.initial_time_idx}")
        print(f"   Max horizon: {config.max_horizon_hours}h ({config.max_horizon_hours//24:.1f} days)")
        print(f"   Time steps: {config.max_steps}")
        
        # Load initial state
        initial_data = self.data_loader.load_features(time_idx=config.initial_time_idx)
        current_state = torch.tensor(initial_data['features'], dtype=torch.float32)
        
        # Clean initial state
        from data.preprocessing import clean_features_for_training
        current_state = clean_features_for_training(current_state)
        
        print(f"   Initial state: {current_state.shape}")
        
        # Run autoregressive prediction
        predictions = {}
        start_time = time.time()
        
        for step in range(config.max_steps):
            step_start = time.time()
            
            try:
                # Make prediction
                wave_pred = self.predictor.predict(current_state)
                
                # Store prediction
                forecast_hours = (step + 1) * config.time_step_hours
                predictions[forecast_hours] = wave_pred.clone()
                
                # Update state for next prediction
                current_state = self._update_state(current_state, wave_pred)
                
                # Progress logging
                if forecast_hours % 24 == 0 or step < 4:  # Daily milestones or first few steps
                    swh_stats = wave_pred[:, 0]
                    elapsed = time.time() - step_start
                    print(f"   t+{forecast_hours:3d}h: SWH {swh_stats.min():.3f}-{swh_stats.max():.3f}m ({elapsed:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ Step {step+1} failed: {e}")
                break
        
        total_time = time.time() - start_time
        
        # Create result
        metadata = {
            'experiment_id': self.predictor.experiment_id,
            'initial_time_idx': config.initial_time_idx,
            'max_horizon_hours': config.max_horizon_hours,
            'time_step_hours': config.time_step_hours,
            'total_steps': len(predictions),
            'forecast_time_seconds': total_time,
            'feature_names': self.predictor.feature_names,
            'wave_indices': self.predictor.wave_indices,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Forecast complete: {len(predictions)} steps in {total_time:.1f}s")
        
        return ForecastResult(predictions=predictions, metadata=metadata)
    
    def _update_state(self, current_state: torch.Tensor, wave_prediction: torch.Tensor) -> torch.Tensor:
        """
        Update state for next autoregressive step
        
        Args:
            current_state: Current feature state [num_nodes, num_features]
            wave_prediction: Latest wave predictions [num_nodes, 3]
            
        Returns:
            Updated state for next prediction
        """
        new_state = current_state.clone()
        
        # Update wave variables with predictions
        if len(self.predictor.wave_indices) >= 3 and wave_prediction.shape[1] >= 3:
            # Update specific wave indices
            for i, wave_idx in enumerate(self.predictor.wave_indices[:3]):
                if i < wave_prediction.shape[1]:
                    new_state[:, wave_idx] = wave_prediction[:, i]
        elif current_state.shape[1] >= 3:
            # Fallback: update last 3 features
            new_state[:, -3:] = wave_prediction[:, :3]
        
        return new_state


class AutoregressiveEvaluator:
    """Evaluates autoregressive forecasts against ground truth"""
    
    def __init__(self, forecaster: AutoregressiveForecaster):
        """
        Args:
            forecaster: Configured AutoregressiveForecaster
        """
        self.forecaster = forecaster
    
    def evaluate(self, forecast_result: ForecastResult) -> ForecastResult:
        """
        Evaluate forecast against ground truth
        
        Args:
            forecast_result: Result from AutoregressiveForecaster.forecast()
            
        Returns:
            ForecastResult with added performance metrics
        """
        print(f"📊 Evaluating forecast against ground truth...")
        
        initial_time = forecast_result.metadata['initial_time_idx']
        time_step_hours = forecast_result.metadata['time_step_hours']
        
        performance = {}
        
        for forecast_hours, pred_waves in forecast_result.predictions.items():
            try:
                # Calculate target time
                forecast_steps = forecast_hours // time_step_hours
                target_time = initial_time + forecast_steps
                
                # Load ground truth
                target_data = self.forecaster.data_loader.load_features(time_idx=target_time)
                target_features = torch.tensor(target_data['features'], dtype=torch.float32)
                
                # Extract wave variables using known indices
                wave_indices = self.forecaster.predictor.wave_indices
                if len(wave_indices) >= 3:
                    true_waves = target_features[:, wave_indices[:3]]
                else:
                    # Fallback: assume last 3 features are waves
                    true_waves = target_features[:, -3:]
                
                # Calculate metrics
                metrics = self._calculate_metrics(pred_waves, true_waves)
                metrics['forecast_hours'] = forecast_hours
                metrics['target_time_idx'] = target_time
                
                performance[forecast_hours] = metrics
                
                # Progress logging for key milestones
                if forecast_hours in [6, 24, 48, 72, 120, 168]:
                    rmse = metrics.get('swh_rmse', 0)
                    bias = metrics.get('swh_bias', 0)
                    corr = metrics.get('swh_correlation', 0)
                    days = forecast_hours // 24
                    hours = forecast_hours % 24
                    time_str = f"{days}d{hours:02d}h" if days > 0 else f"{hours:2d}h"
                    print(f"   t+{time_str}: RMSE={rmse:.4f}m, Bias={bias:+.3f}m, Corr={corr:.3f}")
                
            except Exception as e:
                print(f"   ⚠️  t+{forecast_hours}h: Evaluation failed - {e}")
        
        # Update result with performance
        forecast_result.performance = performance
        print(f"✅ Evaluation complete: {len(performance)} horizons assessed")
        
        return forecast_result
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate performance metrics for wave predictions"""
        
        metrics = {}
        wave_names = ['swh', 'mwd', 'mwp']
        
        for i, name in enumerate(wave_names):
            if i >= predictions.shape[1] or i >= targets.shape[1]:
                continue
                
            pred_vals = predictions[:, i]
            true_vals = targets[:, i]
            
            # Remove invalid values
            valid_mask = ~(torch.isnan(pred_vals) | torch.isnan(true_vals) | 
                          torch.isinf(pred_vals) | torch.isinf(true_vals))
            
            if valid_mask.sum() == 0:
                continue
                
            pred_clean = pred_vals[valid_mask]
            true_clean = true_vals[valid_mask]
            
            # Basic metrics
            rmse = torch.sqrt(torch.mean((pred_clean - true_clean)**2))
            mae = torch.mean(torch.abs(pred_clean - true_clean))
            bias = torch.mean(pred_clean - true_clean)
            
            # Correlation
            if len(pred_clean) > 1:
                pred_centered = pred_clean - pred_clean.mean()
                true_centered = true_clean - true_clean.mean()
                correlation = torch.sum(pred_centered * true_centered) / (
                    torch.sqrt(torch.sum(pred_centered**2)) * torch.sqrt(torch.sum(true_centered**2)) + 1e-8
                )
            else:
                correlation = torch.tensor(0.0)
            
            # Store metrics
            metrics[f'{name}_rmse'] = rmse.item()
            metrics[f'{name}_mae'] = mae.item()
            metrics[f'{name}_bias'] = bias.item()
            metrics[f'{name}_correlation'] = correlation.item()
            metrics[f'{name}_valid_points'] = valid_mask.sum().item()
        
        return metrics
    
    def plot_performance(self, forecast_result: ForecastResult, save_path: Optional[str] = None):
        """Create comprehensive performance plots"""
        
        if forecast_result.performance is None:
            print("⚠️  No performance data available for plotting")
            return
        
        print(f"📊 Creating performance plots...")
        
        # Extract data
        horizons = sorted(forecast_result.performance.keys())
        rmse_values = [forecast_result.performance[h].get('swh_rmse', 0) for h in horizons]
        mae_values = [forecast_result.performance[h].get('swh_mae', 0) for h in horizons]
        bias_values = [forecast_result.performance[h].get('swh_bias', 0) for h in horizons]
        corr_values = [forecast_result.performance[h].get('swh_correlation', 0) for h in horizons]
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        exp_id = forecast_result.metadata['experiment_id']
        fig.suptitle(f'7-Day Autoregressive Wave Forecasting Performance\n{exp_id}', fontsize=16)
        
        # RMSE over time
        axes[0, 0].plot(horizons, rmse_values, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=0.21, color='green', linestyle='--', alpha=0.7, label='Training (6h)')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good (0.5m)')
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Limit (1.0m)')
        axes[0, 0].set_xlabel('Forecast Horizon (hours)')
        axes[0, 0].set_ylabel('SWH RMSE (m)')
        axes[0, 0].set_title('Error Growth')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(horizons, mae_values, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Forecast Horizon (hours)')
        axes[0, 1].set_ylabel('SWH MAE (m)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation
        axes[0, 2].plot(horizons, corr_values, 'purple', linewidth=2, marker='o', markersize=6)
        axes[0, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good')
        axes[0, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair')
        axes[0, 2].set_xlabel('Forecast Horizon (hours)')
        axes[0, 2].set_ylabel('SWH Correlation')
        axes[0, 2].set_title('Correlation vs Truth')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Bias
        axes[1, 0].plot(horizons, bias_values, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel('Forecast Horizon (hours)')
        axes[1, 0].set_ylabel('SWH Bias (m)')
        axes[1, 0].set_title('Prediction Bias')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance degradation
        baseline_rmse = 0.21
        degradation = [rmse / baseline_rmse for rmse in rmse_values]
        axes[1, 1].plot(horizons, degradation, 'orange', linewidth=2, marker='o', markersize=6)
        axes[1, 1].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Baseline')
        axes[1, 1].axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='2x')
        axes[1, 1].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5x')
        axes[1, 1].set_xlabel('Forecast Horizon (hours)')
        axes[1, 1].set_ylabel('Error Multiplier')
        axes[1, 1].set_title('Performance Degradation')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Useful forecast window
        skill_threshold = 0.8
        useful = [1 if rmse <= skill_threshold else 0 for rmse in rmse_values]
        colors = ['green' if u else 'red' for u in useful]
        
        axes[1, 2].bar([h/24 for h in horizons], useful, color=colors, alpha=0.7)
        axes[1, 2].set_xlabel('Forecast Horizon (days)')
        axes[1, 2].set_ylabel('Useful (RMSE ≤ 0.8m)')
        axes[1, 2].set_title('Forecast Skill Window')
        axes[1, 2].set_ylim(0, 1.2)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Plot saved: {save_path}")
        
        plt.show()
    
    def save_results(self, forecast_result: ForecastResult, output_dir: str = "outputs/forecasts"):
        """Save forecast results to JSON"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exp_id = forecast_result.metadata['experiment_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_{exp_id}_{timestamp}.json"
        filepath = output_path / filename
        
        # Prepare data for JSON serialization
        save_data = {
            'metadata': forecast_result.metadata,
            'performance': forecast_result.performance,
            'forecast_horizons': list(forecast_result.predictions.keys()),
            'summary': self._generate_summary(forecast_result)
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved: {filepath}")
        return str(filepath)
    
    def _generate_summary(self, forecast_result: ForecastResult) -> Dict[str, any]:
        """Generate summary statistics"""
        
        if not forecast_result.performance:
            return {}
        
        horizons = sorted(forecast_result.performance.keys())
        rmse_values = [forecast_result.performance[h].get('swh_rmse', 0) for h in horizons]
        
        # Key milestones
        milestones = {}
        for h in [6, 24, 48, 72, 120, 168]:
            if h in forecast_result.performance:
                milestones[f'{h}h'] = {
                    'rmse': forecast_result.performance[h].get('swh_rmse', 0),
                    'bias': forecast_result.performance[h].get('swh_bias', 0),
                    'correlation': forecast_result.performance[h].get('swh_correlation', 0)
                }
        
        # Overall assessment
        final_rmse = rmse_values[-1] if rmse_values else 0
        if final_rmse <= 0.8:
            assessment = "excellent"
        elif final_rmse <= 1.2:
            assessment = "good"
        elif final_rmse <= 2.0:
            assessment = "acceptable"
        else:
            assessment = "poor"
        
        # Useful horizon (where RMSE < 0.8m)
        useful_horizon = 0
        for h, rmse in zip(horizons, rmse_values):
            if rmse <= 0.8:
                useful_horizon = h
        
        return {
            'max_horizon_hours': max(horizons) if horizons else 0,
            'max_horizon_days': max(horizons) / 24 if horizons else 0,
            'final_rmse': final_rmse,
            'useful_horizon_hours': useful_horizon,
            'useful_horizon_days': useful_horizon / 24,
            'assessment': assessment,
            'milestones': milestones,
            'degradation_factor': final_rmse / 0.21 if final_rmse > 0 else 0
        }


# Convenience function for quick usage
def run_full_evaluation(checkpoint_path: str, config: Optional[ForecastConfig] = None) -> ForecastResult:
    """
    Convenience function to run complete 7-day evaluation
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Optional forecast configuration
        
    Returns:
        Complete forecast result with performance evaluation
    """
    
    if config is None:
        config = ForecastConfig()
    
    print(f"🌊 Running full 7-day autoregressive evaluation")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Max horizon: {config.max_horizon_hours}h")
    
    # Setup components
    predictor = WavePredictor.from_checkpoint(checkpoint_path)
    
    # Setup data loader (reuse the default setup)
    from data.loaders import ERA5DataManager, GEBCODataManager
    from data.preprocessing import MultiResolutionInterpolator
    from data.datasets import MeshDataLoader
    from mesh.icosahedral import IcosahedralMesh
    from config.base import DataConfig, MeshConfig
    
    data_config = DataConfig()
    mesh_config = MeshConfig(refinement_level=5)
    
    era5_manager = ERA5DataManager(data_config)
    gebco_manager = GEBCODataManager(data_config)
    
    first_file = era5_manager.available_files['atmospheric'][0]
    filename = Path(first_file).stem
    year_month = filename.split('_')[-1]
    year, month = int(year_month[:4]), int(year_month[4:6])
    
    era5_atmo, era5_waves = era5_manager.load_month_data(year, month)
    gebco_data = gebco_manager.load_bathymetry()
    
    mesh = IcosahedralMesh(mesh_config)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
    mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
    
    # Create components
    forecaster = AutoregressiveForecaster(predictor, mesh_loader)
    evaluator = AutoregressiveEvaluator(forecaster)
    
    # Run forecast
    forecast_result = forecaster.forecast(config)
    
    # Evaluate
    forecast_result = evaluator.evaluate(forecast_result)
    
    # Generate plots
    output_dir = Path("outputs/forecasts")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"forecast_{predictor.experiment_id}.png"
    evaluator.plot_performance(forecast_result, str(plot_path))
    
    # Save results
    evaluator.save_results(forecast_result)
    
    return forecast_result