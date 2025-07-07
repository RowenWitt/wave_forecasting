# =============================================================================
# AUTOREGRESSIVE WAVE FORECASTING SYSTEM
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass


# =============================================================================
# 1. AUTOREGRESSIVE PREDICTOR
# =============================================================================

class AutoregressiveWavePredictor:
    """
    Uses your trained spatial model to make multi-step forecasts
    """
    
    def __init__(self, spatial_model, mesh_loader, edge_index, edge_attr, 
                 feature_names: List[str]):
        """
        spatial_model: Your trained SpatialWaveGNN with 0.21m RMSE
        mesh_loader: For getting initial conditions
        edge_index, edge_attr: Graph connectivity 
        feature_names: List of feature names in order (e.g., ['u10', 'v10', 'msl', 'swh', 'mwd', 'mwp', ...])
        """
        self.model = spatial_model
        self.mesh_loader = mesh_loader
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.feature_names = feature_names
        
        # Identify which features are wave variables
        self.wave_indices = self._find_wave_indices()
        
        print(f"🔄 Autoregressive Predictor Ready:")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Wave indices: {self.wave_indices} ({[feature_names[i] for i in self.wave_indices]})")
    
    def _find_wave_indices(self):
        """Find indices of wave variables in feature vector"""
        wave_vars = ['swh', 'mwd', 'mwp']
        indices = []
        
        for wave_var in wave_vars:
            for i, feature_name in enumerate(self.feature_names):
                if wave_var in feature_name.lower():
                    indices.append(i)
                    break
        
        if len(indices) != 3:
            print(f"⚠️  Warning: Expected 3 wave variables, found {len(indices)}")
            print(f"   Available features: {self.feature_names}")
        
        return indices
    
    def predict_sequence(self, initial_time_idx: int, 
                        forecast_steps: List[int] = [1, 2, 3, 4, 8, 12],
                        time_step_hours: int = 6) -> Dict[int, torch.Tensor]:
        """
        Make autoregressive predictions
        
        Args:
            initial_time_idx: Starting time index in your data
            forecast_steps: List of steps ahead to predict (1 step = 6 hours)
            time_step_hours: Hours per timestep (6 for ERA5)
        
        Returns:
            Dictionary mapping forecast_hours → wave predictions [nodes, 3]
        """
        
        print(f"🔮 Autoregressive Forecast:")
        print(f"   Initial time: {initial_time_idx}")
        print(f"   Steps: {forecast_steps} (max {max(forecast_steps) * time_step_hours}h ahead)")
        
        # Load initial conditions
        initial_data = self.mesh_loader.load_features(time_idx=initial_time_idx)
        current_state = torch.tensor(initial_data['features'], dtype=torch.float32)
        
        # Clean initial state
        from data.preprocessing import clean_features_for_training
        current_state = clean_features_for_training(current_state)
        
        print(f"   Initial state: {current_state.shape}")
        print(f"   Initial SWH range: {current_state[:, self.wave_indices[0]].min():.3f}-{current_state[:, self.wave_indices[0]].max():.3f}m")
        
        self.model.eval()
        predictions = {}
        prediction_history = []  # Track how state evolves
        
        with torch.no_grad():
            for step in range(max(forecast_steps)):
                step_start_time = time.time()
                
                # Predict next wave state
                wave_prediction = self.model(current_state, self.edge_index, self.edge_attr)
                
                # Store prediction if it's a requested step
                if (step + 1) in forecast_steps:
                    forecast_hours = (step + 1) * time_step_hours
                    predictions[forecast_hours] = wave_prediction.clone()
                    
                    swh_range = wave_prediction[:, 0]
                    print(f"   t+{forecast_hours:2d}h: SWH {swh_range.min():.3f}-{swh_range.max():.3f}m "
                          f"({time.time() - step_start_time:.2f}s)")
                
                # Update current state for next iteration
                # STRATEGY: Only update wave variables, keep atmospheric/bathy features
                current_state = self._update_state_for_next_step(current_state, wave_prediction, step)
                
                # Track evolution
                prediction_history.append({
                    'step': step + 1,
                    'hours': (step + 1) * time_step_hours,
                    'wave_prediction': wave_prediction.clone(),
                    'full_state': current_state.clone()
                })
        
        print(f"✅ Autoregressive forecast complete: {len(predictions)} horizons")
        
        # Store history for analysis
        self.last_prediction_history = prediction_history
        
        return predictions
    
    def _update_state_for_next_step(self, current_state: torch.Tensor, 
                                   wave_prediction: torch.Tensor, step: int) -> torch.Tensor:
        """
        Update state for next autoregressive step
        
        STRATEGY OPTIONS:
        1. Simple: Only update wave variables, keep others constant
        2. Realistic: Try to evolve atmospheric variables too  
        3. Hybrid: Update waves, decay atmospheric toward climatology
        """
        
        new_state = current_state.clone()
        
        # SIMPLE STRATEGY (start with this):
        # Update wave variables with predictions, keep everything else the same
        new_state[:, self.wave_indices] = wave_prediction
        
        # OPTIONAL: Add some realistic evolution
        # For longer forecasts, might want to evolve atmospheric variables
        if step >= 4:  # After 24 hours, start evolving atmospheric vars
            atmo_indices = self._find_atmospheric_indices()
            # Simple decay toward mean (crude but helps prevent unrealistic drift)
            decay_factor = 0.95
            for idx in atmo_indices:
                mean_val = current_state[:, idx].mean()
                new_state[:, idx] = decay_factor * new_state[:, idx] + (1 - decay_factor) * mean_val
        
        return new_state
    
    def _find_atmospheric_indices(self):
        """Find atmospheric variable indices"""
        atmo_vars = ['u10', 'v10', 'msl']  # Wind and pressure
        indices = []
        
        for atmo_var in atmo_vars:
            for i, feature_name in enumerate(self.feature_names):
                if atmo_var in feature_name.lower():
                    indices.append(i)
                    break
        
        return indices



# =============================================================================
# 3. VISUALIZATION TOOLS
# =============================================================================

def plot_autoregressive_performance(summary_results: Dict[str, any], 
                                   save_path: Optional[str] = None):
    """Plot autoregressive forecast performance vs horizon"""
    
    horizons = sorted(summary_results.keys())
    
    # Extract metrics
    rmse_means = [summary_results[h].get('swh_rmse_mean', 0) for h in horizons]
    rmse_stds = [summary_results[h].get('swh_rmse_std', 0) for h in horizons]
    corr_means = [summary_results[h].get('swh_corr_mean', 0) for h in horizons]
    bias_means = [summary_results[h].get('swh_bias_mean', 0) for h in horizons]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE vs Horizon
    axes[0, 0].errorbar(horizons, rmse_means, yerr=rmse_stds, 
                       marker='o', capsize=5, linewidth=2)
    axes[0, 0].set_xlabel('Forecast Horizon (hours)')
    axes[0, 0].set_ylabel('SWH RMSE (m)')
    axes[0, 0].set_title('Wave Height Error vs Forecast Horizon')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add benchmark lines
    axes[0, 0].axhline(y=0.21, color='green', linestyle='--', alpha=0.7, label='Spatial Model (6h)')
    axes[0, 0].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='NOAA WaveWatch III')
    axes[0, 0].legend()
    
    # Correlation vs Horizon
    axes[0, 1].plot(horizons, corr_means, marker='o', linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Forecast Horizon (hours)')
    axes[0, 1].set_ylabel('SWH Correlation')
    axes[0, 1].set_title('Wave Height Correlation vs Horizon')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bias vs Horizon
    axes[1, 0].plot(horizons, bias_means, marker='o', linewidth=2, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Forecast Horizon (hours)')
    axes[1, 0].set_ylabel('SWH Bias (m)')
    axes[1, 0].set_title('Wave Height Bias vs Horizon')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Skill degradation
    skill_scores = []
    for h in horizons:
        rmse = summary_results[h].get('swh_rmse_mean', 1.0)
        # Simple skill score relative to 6h performance
        skill = max(0, 1 - (rmse / 0.21))
        skill_scores.append(skill)
    
    axes[1, 1].plot(horizons, skill_scores, marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Forecast Horizon (hours)')
    axes[1, 1].set_ylabel('Relative Skill Score')
    axes[1, 1].set_title('Skill Degradation vs Horizon')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Autoregressive Wave Forecast Performance', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

