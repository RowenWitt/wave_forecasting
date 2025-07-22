#!/usr/bin/env python3
"""
Global Wave Model Evaluation Script
Comprehensive evaluation with regional performance analysis
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np
import torch
import xarray as xr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from global_wave_model_v1 import GlobalWaveConfig, VariableSpecificNormalizer, CircularNormalizer

warnings.filterwarnings('ignore')

# Import components from the training script
# Note: In practice, you'd import from your training module
# For now, we'll redefine the necessary components

class CircularMetrics:
    """Compute metrics for circular variables like wave direction"""
    
    @staticmethod
    def angular_distance(pred_deg, true_deg):
        """Compute angular distance in degrees"""
        diff = pred_deg - true_deg
        # Wrap to [-180, 180]
        diff = (diff + 180) % 360 - 180
        return np.abs(diff)
    
    @staticmethod
    def circular_mae(pred_deg, true_deg):
        """Mean absolute error for circular variables"""
        return np.mean(CircularMetrics.angular_distance(pred_deg, true_deg))
    
    @staticmethod
    def circular_rmse(pred_deg, true_deg):
        """RMSE for circular variables"""
        angular_dist = CircularMetrics.angular_distance(pred_deg, true_deg)
        return np.sqrt(np.mean(angular_dist ** 2))

class RegionalEvaluator:
    """Evaluate model performance by geographic region"""
    
    def __init__(self):
        # Define evaluation regions
        self.regions = {
            # Major ocean basins
            'North Atlantic': {'lat': (20, 70), 'lon': (-80, 0)},
            'South Atlantic': {'lat': (-60, 20), 'lon': (-70, 20)},
            'North Pacific': {'lat': (20, 60), 'lon': (120, 240)},
            'South Pacific': {'lat': (-60, 20), 'lon': (150, 290)},
            'Indian Ocean': {'lat': (-60, 30), 'lon': (20, 120)},
            'Southern Ocean': {'lat': (-90, -45), 'lon': (0, 360)},
            'Arctic Ocean': {'lat': (70, 90), 'lon': (0, 360)},
            'Mediterranean': {'lat': (30, 45), 'lon': (-5, 40)},
            
            # Specific forecast regions
            'Tropical Atlantic': {'lat': (-20, 20), 'lon': (-60, 20)},
            'Tropical Pacific': {'lat': (-20, 20), 'lon': (120, 280)},
            'North Sea': {'lat': (50, 62), 'lon': (-5, 10)},
            'Caribbean': {'lat': (10, 30), 'lon': (-90, -60)},
            
            # Latitude bands
            'Tropics': {'lat': (-23.5, 23.5), 'lon': (0, 360)},
            'Mid-Latitudes North': {'lat': (23.5, 60), 'lon': (0, 360)},
            'Mid-Latitudes South': {'lat': (-60, -23.5), 'lon': (0, 360)},
            'High-Latitudes North': {'lat': (60, 90), 'lon': (0, 360)},
            'High-Latitudes South': {'lat': (-90, -60), 'lon': (0, 360)},
        }
    
    def get_region_mask(self, lats, lons, region_name):
        """Get boolean mask for a specific region"""
        if region_name not in self.regions:
            raise ValueError(f"Unknown region: {region_name}")
        
        region = self.regions[region_name]
        lat_bounds = region['lat']
        lon_bounds = region['lon']
        
        # Handle longitude wraparound
        if lon_bounds[0] > lon_bounds[1]:  # Crosses 0 meridian
            lon_mask = (lons >= lon_bounds[0]) | (lons <= lon_bounds[1])
        else:
            lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])
        
        lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
        
        return lat_mask & lon_mask
    
    def evaluate_by_region(self, predictions, targets, lats, lons, metrics_dict):
        """Compute metrics for each geographic region"""
        regional_results = {}
        
        for region_name in self.regions:
            mask = self.get_region_mask(lats, lons, region_name)
            
            if mask.sum() == 0:  # No points in region
                continue
            
            regional_results[region_name] = {}
            
            # Apply mask and compute metrics
            for var_idx, var_name in enumerate(['swh', 'mwd', 'mwp']):
                pred_var = predictions[:, var_idx][mask]
                true_var = targets[:, var_idx][mask]
                
                if var_name == 'mwd':
                    # Special handling for circular variable
                    mae = CircularMetrics.circular_mae(pred_var, true_var)
                    rmse = CircularMetrics.circular_rmse(pred_var, true_var)
                else:
                    mae = mean_absolute_error(true_var, pred_var)
                    rmse = np.sqrt(mean_squared_error(true_var, pred_var))
                
                regional_results[region_name][f'{var_name}_mae'] = mae
                regional_results[region_name][f'{var_name}_rmse'] = rmse
                
                # Compute relative error for non-directional variables
                if var_name != 'mwd':
                    mean_true = np.mean(true_var)
                    if mean_true > 0:
                        regional_results[region_name][f'{var_name}_mape'] = mae / mean_true * 100
            
            regional_results[region_name]['n_points'] = int(mask.sum())
        
        return regional_results

class GlobalWaveEvaluator:
    """Main evaluation class for global wave model"""
    
    def __init__(self, checkpoint_path: str, data_path: str, output_dir: str = "evaluation_results"):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract components
        self.config = self.checkpoint['config']
        self.feature_normalizer = self.checkpoint['feature_normalizer']
        self.target_normalizer = self.checkpoint['target_normalizer']
        self.edge_index = self.checkpoint['edge_index']
        self.edge_attr = self.checkpoint['edge_attr']
        self.mesh_vertices = self.checkpoint['mesh_vertices']
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Regional evaluator
        self.regional_evaluator = RegionalEvaluator()
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"🖥️  Device: {self.device}")
    
    def load_model(self):
        """Load model from checkpoint with multiscale support"""
    
        # Check if this is a multiscale model
        is_multiscale = 'edge_slices' in self.checkpoint
        
        if is_multiscale:
            print("   🔗 Detected multiscale model")
            # Import multiscale model class
            from global_wave_model_multiscale import MultiscaleGlobalWaveGNN
            
            model = MultiscaleGlobalWaveGNN(self.config)
            model.load_state_dict(self.checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # Store edge slices for forward pass
            self.edge_slices = self.checkpoint.get('edge_slices', {})
        else:
            print("   📍 Detected standard model")
            # Import standard model class
            from global_wave_model_v1 import GlobalWaveGNN
            
            model = GlobalWaveGNN(self.config)
            model.load_state_dict(self.checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.edge_slices = None
        
        return model
    
    def prepare_validation_data(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Prepare validation dataset"""
        print(f"\n📊 Loading validation data from: {self.data_path}")
        
        # Import dataset class
        from global_wave_model_v1 import GlobalWaveDataset, GlobalIcosahedralMesh
        
        # Recreate mesh
        mesh = GlobalIcosahedralMesh(refinement_level=self.config.mesh_refinement_level)
        mesh.vertices = self.mesh_vertices  # Use saved vertices
        
        # Create dataset
        dataset = GlobalWaveDataset(
            data_path=str(self.data_path),
            mesh=mesh,
            config=self.config,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        return dataset, mesh
    
    def evaluate_full_dataset(self, model, dataset, mesh, batch_size: int = 4):
        """Run evaluation on full dataset"""
        print(f"\n🔄 Running full dataset evaluation...")
        
        # Get mesh coordinates
        mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
        
        # Storage for results
        all_predictions = []
        all_targets = []
        
        # Process in batches
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"   Processing batch {batch_idx}/{len(dataloader)}")
                
                # Move to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Normalize inputs
                batch_size, seq_len, num_nodes, num_features = inputs.size()
                inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                inputs_norm = self.feature_normalizer.transform(inputs_flat)
                inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                
                # Forward pass - handle both model types
                edge_index = self.edge_index.to(self.device)
                edge_attr = self.edge_attr.to(self.device)
                
                if self.edge_slices is not None:
                    # Multiscale model
                    predictions = model(inputs, edge_index, edge_attr, self.edge_slices)
                else:
                    # Standard model
                    predictions = model(inputs, edge_index, edge_attr)
                
                # Denormalize predictions and targets
                predictions_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                for i in range(batch_size):
                    # Denormalize predictions
                    pred_norm = predictions_np[i]  # [nodes, 4]
                    pred_denorm = self.target_normalizer.inverse_transform_targets(pred_norm)
                    all_predictions.append(pred_denorm)
                    
                    # Original targets
                    target_orig = targets_np[i]  # [nodes, 3]
                    all_targets.append(target_orig)
        
        # Stack results
        all_predictions = np.array(all_predictions)  # [n_samples, n_nodes, 3]
        all_targets = np.array(all_targets)  # [n_samples, n_nodes, 3]
        
        print(f"✅ Evaluation complete: {all_predictions.shape[0]} samples")
        
        return all_predictions, all_targets, mesh_lats, mesh_lons
    
    def compute_global_metrics(self, predictions, targets):
        """Compute global metrics for each variable"""
        metrics = {}
        
        for var_idx, var_name in enumerate(['swh', 'mwd', 'mwp']):
            pred = predictions[:, :, var_idx].flatten()
            true = targets[:, :, var_idx].flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred) | np.isnan(true))
            pred = pred[valid_mask]
            true = true[valid_mask]
            
            if var_name == 'mwd':
                # Circular metrics
                mae = CircularMetrics.circular_mae(pred, true)
                rmse = CircularMetrics.circular_rmse(pred, true)
            else:
                mae = mean_absolute_error(true, pred)
                rmse = np.sqrt(mean_squared_error(true, pred))
            
            metrics[var_name] = {
                'mae': mae,
                'rmse': rmse,
                'bias': np.mean(pred - true) if var_name != 'mwd' else None,
                'std_error': np.std(pred - true) if var_name != 'mwd' else None,
                'correlation': np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else 0.0
            }
            
            # Add relative metrics for non-directional variables
            if var_name != 'mwd':
                mean_true = np.mean(true)
                if mean_true > 0:
                    metrics[var_name]['mape'] = mae / mean_true * 100
                    metrics[var_name]['nrmse'] = rmse / mean_true * 100
        
        return metrics
    
    def plot_regional_performance(self, regional_results, save_path: Optional[str] = None):
        """Create visualization of regional performance"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color maps for each variable
        cmaps = {'swh': 'Blues_r', 'mwd': 'RdYlBu_r', 'mwp': 'Greens_r'}
        
        for var_idx, var_name in enumerate(['swh', 'mwd', 'mwp']):
            # Extract RMSE for each region
            region_rmse = {}
            for region, metrics in regional_results.items():
                if f'{var_name}_rmse' in metrics:
                    region_rmse[region] = metrics[f'{var_name}_rmse']
            
            if not region_rmse:
                continue
            
            # Create subplot
            ax = fig.add_subplot(gs[var_idx, :2], projection=ccrs.Robinson())
            ax.set_global()
            ax.add_feature(cfeature.LAND, color='lightgray')
            ax.add_feature(cfeature.OCEAN, color='white')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.gridlines(draw_labels=False, alpha=0.3)
            
            # Plot regions with color based on RMSE
            vmin = min(region_rmse.values())
            vmax = max(region_rmse.values())
            
            for region_name, rmse in region_rmse.items():
                if region_name in self.regional_evaluator.regions:
                    region_def = self.regional_evaluator.regions[region_name]
                    lat_bounds = region_def['lat']
                    lon_bounds = region_def['lon']
                    
                    # Create region rectangle
                    if lon_bounds[0] > lon_bounds[1]:  # Crosses dateline
                        # Split into two rectangles
                        rect1 = plt.Rectangle((lon_bounds[0], lat_bounds[0]), 
                                            360 - lon_bounds[0], 
                                            lat_bounds[1] - lat_bounds[0],
                                            transform=ccrs.PlateCarree(),
                                            facecolor=plt.cm.get_cmap(cmaps[var_name])((rmse - vmin) / (vmax - vmin)),
                                            edgecolor='black',
                                            alpha=0.7,
                                            linewidth=0.5)
                        rect2 = plt.Rectangle((0, lat_bounds[0]), 
                                            lon_bounds[1], 
                                            lat_bounds[1] - lat_bounds[0],
                                            transform=ccrs.PlateCarree(),
                                            facecolor=plt.cm.get_cmap(cmaps[var_name])((rmse - vmin) / (vmax - vmin)),
                                            edgecolor='black',
                                            alpha=0.7,
                                            linewidth=0.5)
                        ax.add_patch(rect1)
                        ax.add_patch(rect2)
                    else:
                        rect = plt.Rectangle((lon_bounds[0], lat_bounds[0]), 
                                           lon_bounds[1] - lon_bounds[0], 
                                           lat_bounds[1] - lat_bounds[0],
                                           transform=ccrs.PlateCarree(),
                                           facecolor=plt.cm.get_cmap(cmaps[var_name])((rmse - vmin) / (vmax - vmin)),
                                           edgecolor='black',
                                           alpha=0.7,
                                           linewidth=0.5)
                        ax.add_patch(rect)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmaps[var_name]), 
                                      norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(f'{var_name.upper()} RMSE', fontsize=12)
            
            ax.set_title(f'{var_name.upper()} Regional Performance', fontsize=14, pad=10)
            
            # Create bar chart for top/bottom regions
            ax_bar = fig.add_subplot(gs[var_idx, 2])
            
            # Sort regions by RMSE
            sorted_regions = sorted(region_rmse.items(), key=lambda x: x[1])
            top_5 = sorted_regions[:5]
            bottom_5 = sorted_regions[-5:]
            
            regions = [r[0] for r in top_5] + ['...'] + [r[0] for r in bottom_5]
            values = [r[1] for r in top_5] + [0] + [r[1] for r in bottom_5]
            colors = ['green'] * 5 + ['white'] + ['red'] * 5
            
            y_pos = np.arange(len(regions))
            ax_bar.barh(y_pos, values, color=colors, alpha=0.7)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(regions, fontsize=10)
            ax_bar.set_xlabel(f'{var_name.upper()} RMSE')
            ax_bar.set_title(f'Best/Worst Regions', fontsize=12)
            ax_bar.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Regional Model Performance', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'regional_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scatter_comparisons(self, predictions, targets, save_path: Optional[str] = None):
        """Create scatter plots comparing predictions vs targets"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for var_idx, (var_name, ax) in enumerate(zip(['swh', 'mwd', 'mwp'], axes)):
            # Flatten and sample data
            pred = predictions[:, :, var_idx].flatten()
            true = targets[:, :, var_idx].flatten()
            
            # Remove NaN and sample for plotting
            valid_mask = ~(np.isnan(pred) | np.isnan(true))
            pred = pred[valid_mask]
            true = true[valid_mask]
            
            # Sample if too many points
            if len(pred) > 10000:
                idx = np.random.choice(len(pred), 10000, replace=False)
                pred = pred[idx]
                true = true[idx]
            
            # Create scatter plot
            ax.scatter(true, pred, alpha=0.5, s=1)
            
            # Add diagonal line
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=2)
            
            # Add metrics
            if var_name == 'mwd':
                mae = CircularMetrics.circular_mae(pred, true)
                rmse = CircularMetrics.circular_rmse(pred, true)
            else:
                mae = mean_absolute_error(true, pred)
                rmse = np.sqrt(mean_squared_error(true, pred))
            
            corr = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else 0.0
            
            textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nCorr: {corr:.3f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'True {var_name.upper()}')
            ax.set_ylabel(f'Predicted {var_name.upper()}')
            ax.set_title(f'{var_name.upper()} Predictions vs Truth')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'scatter_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_report(self, global_metrics, regional_results):
        """Save comprehensive metrics report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'checkpoint_path': str(self.checkpoint_path),
            'data_path': str(self.data_path),
            'global_metrics': global_metrics,
            'regional_results': regional_results
        }
        
        # Save as JSON
        with open(self.output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(str(report), f, indent=2)
        
        # Create readable text report
        with open(self.output_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("GLOBAL WAVE MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report['evaluation_timestamp']}\n")
            f.write(f"Checkpoint: {report['checkpoint_path']}\n")
            f.write(f"Data: {report['data_path']}\n\n")
            
            f.write("GLOBAL METRICS\n")
            f.write("-" * 30 + "\n")
            for var_name, metrics in global_metrics.items():
                f.write(f"\n{var_name.upper()}:\n")
                for metric_name, value in metrics.items():
                    if value is not None:
                        f.write(f"  {metric_name}: {value:.4f}\n")
            
            f.write("\n\nREGIONAL PERFORMANCE (Top 5 per variable)\n")
            f.write("-" * 30 + "\n")
            
            for var_name in ['swh', 'mwd', 'mwp']:
                f.write(f"\n{var_name.upper()} - Best Regions:\n")
                
                # Sort regions by RMSE
                region_rmse = []
                for region, metrics in regional_results.items():
                    if f'{var_name}_rmse' in metrics:
                        region_rmse.append((region, metrics[f'{var_name}_rmse']))
                
                region_rmse.sort(key=lambda x: x[1])
                
                for i, (region, rmse) in enumerate(region_rmse[:5]):
                    mae = regional_results[region][f'{var_name}_mae']
                    f.write(f"  {i+1}. {region}: RMSE={rmse:.4f}, MAE={mae:.4f}\n")
        
        print(f"\n📊 Evaluation report saved to: {self.output_dir}")
    
    def run_complete_evaluation(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Run complete evaluation pipeline"""
        print("\n🚀 Starting Global Wave Model Evaluation")
        print("=" * 50)
        
        # Load model
        model = self.load_model()
        
        # Prepare data
        dataset, mesh = self.prepare_validation_data(start_idx, end_idx)
        
        # Run evaluation
        predictions, targets, mesh_lats, mesh_lons = self.evaluate_full_dataset(
            model, dataset, mesh
        )
        
        # Compute global metrics
        print("\n📊 Computing global metrics...")
        global_metrics = self.compute_global_metrics(predictions, targets)
        
        print("\nGlobal Performance:")
        for var_name, metrics in global_metrics.items():
            print(f"\n{var_name.upper()}:")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"  {metric_name}: {value:.4f}")
        
        # Compute regional metrics
        print("\n🌍 Computing regional metrics...")
        
        # Flatten predictions and targets for regional analysis
        all_pred_flat = predictions.reshape(-1, 3)
        all_true_flat = targets.reshape(-1, 3)
        all_lats_flat = np.tile(mesh_lats, predictions.shape[0])
        all_lons_flat = np.tile(mesh_lons, predictions.shape[0])
        
        regional_results = self.regional_evaluator.evaluate_by_region(
            all_pred_flat, all_true_flat, all_lats_flat, all_lons_flat, global_metrics
        )
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        self.plot_regional_performance(regional_results)
        self.plot_scatter_comparisons(predictions, targets)
        
        # Save report
        self.save_metrics_report(global_metrics, regional_results)
        
        print("\n✅ Evaluation complete!")
        print(f"   Results saved to: {self.output_dir}")
        
        return global_metrics, regional_results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Global Wave Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to validation data (NetCDF)')
    parser.add_argument('--output_dir', type=str, default='experiments/global_wave_v1/evals',
                       help='Output directory for results')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start index for validation data')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End index for validation data')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GlobalWaveEvaluator(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.run_complete_evaluation(
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )


if __name__ == "__main__":
    main()