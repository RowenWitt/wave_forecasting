#!/usr/bin/env python3
"""
Unified Model Evaluation Framework
Compares SpatioTemporal, SpatioTemporalCircular, EnhancedPhysics, and ProductionSpatial models
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing components
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import DataConfig, MeshConfig
from enhanced_phyics_wave_model import EnhancedPhysicsConfig
from spatiotemporal_with_circular_mwd import SpatioTemporalConfig

@dataclass
class ModelConfig:
    """Configuration for each model type"""
    name: str
    model_path: str
    input_features: int
    output_format: str  # 'linear_mwd' or 'circular_mwd'
    has_multi_step: bool = True
    requires_edge_data: bool = True
    
@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    test_start_time: int = 400  # Start of test period
    test_end_time: int = 500    # End of test period
    sequence_length: int = 6
    prediction_horizon: int = 4
    batch_size: int = 8
    num_test_sequences: int = 50  # Configurable test size

class ModelWrapper:
    """Wrapper to standardize model interfaces"""
    
    def __init__(self, model_config: ModelConfig, device: torch.device):
        self.config = model_config
        self.device = device
        self.model = None
        self.normalizer = None
        self.target_normalizer = None
        self.edge_index = None
        self.edge_attr = None
        
    def load_model(self):
        """Load model and associated components"""
        print(f"Loading {self.config.name} from {self.config.model_path}")
        
        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
            
            # Extract components based on checkpoint structure
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                self.normalizer = checkpoint.get('feature_normalizer')
                self.target_normalizer = checkpoint.get('target_normalizer')
                self.edge_index = checkpoint.get('edge_index')
                self.edge_attr = checkpoint.get('edge_attr')
            else:
                # Handle different checkpoint formats
                model_state = checkpoint
                
            # Load model architecture based on type
            self.model = self._create_model_architecture()
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"  ✅ Loaded {self.config.name} successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load {self.config.name}: {e}")
            return False
    
    def _create_model_architecture(self):
        """Create model architecture based on config"""
        # This needs to be customized based on your actual model classes
        if 'Enhanced' in self.config.name:
            from enhanced_phyics_wave_model import EnhancedSpatioTemporalWaveGNN, EnhancedPhysicsConfig
            config = EnhancedPhysicsConfig()
            return EnhancedSpatioTemporalWaveGNN(config)
        elif 'Circular' in self.config.name:
            from spatiotemporal_with_circular_mwd import SpatioTemporalWaveGNN
            # Need to determine config from checkpoint
            return SpatioTemporalWaveGNN(input_features=self.config.input_features)
        else:
            # Default spatiotemporal or production spatial
            from new_architecture_test import SpatioTemporalWaveGNN
            return SpatioTemporalWaveGNN(input_features=self.config.input_features)
    
    def prepare_input(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Standardize input format for this model"""
        
        # Handle feature dimension matching
        batch_size, seq_len, num_nodes, raw_features = raw_input.shape
        
        if raw_features != self.config.input_features:
            if raw_features > self.config.input_features:
                # Truncate features
                processed_input = raw_input[:, :, :, :self.config.input_features]
            else:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len, num_nodes, 
                                    self.config.input_features - raw_features,
                                    device=raw_input.device)
                processed_input = torch.cat([raw_input, padding], dim=-1)
        else:
            processed_input = raw_input
        
        # Apply normalization if available
        if self.normalizer is not None:
            flat_input = processed_input.view(-1, self.config.input_features).cpu().numpy()
            normalized = self.normalizer.transform(flat_input)
            processed_input = torch.tensor(normalized, dtype=torch.float32, device=self.device)
            processed_input = processed_input.view(batch_size, seq_len, num_nodes, self.config.input_features)
        
        return processed_input
    
    def predict(self, input_tensor: torch.Tensor, multi_step: bool = True) -> torch.Tensor:
        """Make prediction with this model"""
        
        processed_input = self.prepare_input(input_tensor)
        
        with torch.no_grad():
            if self.config.requires_edge_data and self.edge_index is not None:
                predictions = self.model(processed_input, self.edge_index.to(self.device), 
                                       self.edge_attr.to(self.device), multi_step=multi_step)
            else:
                predictions = self.model(processed_input)
        
        return predictions
    
    def postprocess_output(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert model output to standard format [SWH, MWD_degrees, MWP]"""
        
        if self.config.output_format == 'circular_mwd':
            # Convert [SWH, MWD_cos, MWD_sin, MWP] to [SWH, MWD_degrees, MWP]
            swh = predictions[..., 0]
            mwd_cos = predictions[..., 1]
            mwd_sin = predictions[..., 2]
            mwp = predictions[..., 3]
            
            # Convert circular to degrees
            mwd_rad = torch.atan2(mwd_sin, mwd_cos)
            mwd_degrees = torch.rad2deg(mwd_rad) % 360
            
            return torch.stack([swh, mwd_degrees, mwp], dim=-1)
        else:
            # Already in linear format, just ensure 3 outputs
            if predictions.shape[-1] >= 3:
                return predictions[..., :3]
            else:
                # Pad if needed
                padding = torch.zeros(*predictions.shape[:-1], 3 - predictions.shape[-1], 
                                    device=predictions.device)
                return torch.cat([predictions, padding], dim=-1)

class UnifiedEvaluator:
    """Unified evaluation framework for all models"""
    
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"🔧 Unified Evaluator initialized on {self.device}")
        
        # Results storage
        self.results = {}
        self.test_data = None
        
    def setup_test_data(self):
        """Setup standardized test dataset"""
        print("📊 Setting up standardized test dataset...")
        
        # Load data components (same as training)
        data_config = DataConfig()
        mesh_config = MeshConfig(refinement_level=5)
        
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load test period data (2021 for consistency)
        era5_atmo, era5_waves = era5_manager.load_month_data(2022, 3)  # December 2021
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh and interpolator
        mesh = IcosahedralMesh(mesh_config)
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Create test sequences
        test_sequences = []
        for t in range(self.config.test_start_time, 
                      min(self.config.test_end_time, len(era5_atmo.valid_time) - 10)):
            try:
                # Input sequence (standardized to 11 features for compatibility)
                input_features = []
                for i in range(self.config.sequence_length):
                    features_data = mesh_loader.load_features(time_idx=t + i)
                    features = torch.tensor(features_data['features'], dtype=torch.float32)
                    features = torch.nan_to_num(features, nan=0.0)
                    input_features.append(features)
                
                # Target sequence
                target_features = []
                for i in range(self.config.prediction_horizon):
                    features_data = mesh_loader.load_features(time_idx=t + self.config.sequence_length + i)
                    targets = torch.tensor(features_data['features'][:, [3, 4, 5]], dtype=torch.float32)
                    targets = torch.nan_to_num(targets, nan=0.0)
                    target_features.append(targets)
                
                input_tensor = torch.stack(input_features, dim=0)
                target_tensor = torch.stack(target_features, dim=0)
                
                test_sequences.append({
                    'input': input_tensor,
                    'target': target_tensor,
                    'timestep': t
                })
                
                if len(test_sequences) >= self.config.num_test_sequences:
                    break
                    
            except Exception as e:
                print(e)
                continue
        
        self.test_data = test_sequences
        print(f"  ✅ Created {len(test_sequences)} test sequences")
        
        return mesh_loader
    
    def evaluate_model(self, model_wrapper: ModelWrapper) -> Dict[str, float]:
        """Evaluate a single model on test data"""
        
        if not model_wrapper.load_model():
            return {'error': 'Failed to load model'}
        
        print(f"🧪 Evaluating {model_wrapper.config.name}...")
        
        all_predictions = []
        all_targets = []
        step_errors = {f'step_{i+1}': [] for i in range(self.config.prediction_horizon)}
        
        for seq_idx, sequence in enumerate(self.test_data):
            try:
                # Prepare batch
                input_batch = sequence['input'].unsqueeze(0).to(self.device)  # [1, seq, nodes, features]
                target_batch = sequence['target']  # [horizon, nodes, 3]
                
                # Predict
                predictions = model_wrapper.predict(input_batch, multi_step=True)
                
                # Postprocess to standard format
                pred_standard = model_wrapper.postprocess_output(predictions)
                
                # Handle different prediction shapes
                if pred_standard.dim() == 2:  # [nodes, features] - single step
                    pred_standard = pred_standard.unsqueeze(0)  # [1, nodes, features]
                elif pred_standard.dim() == 3 and pred_standard.shape[0] == 1:  # [1, nodes, features]
                    pass  # Already correct
                elif pred_standard.dim() == 3:  # [nodes, horizon*features] - multi-step flattened
                    nodes, total_feat = pred_standard.shape[1], pred_standard.shape[2]
                    pred_standard = pred_standard.view(1, nodes, self.config.prediction_horizon, 3)
                    pred_standard = pred_standard.transpose(1, 2)  # [1, horizon, nodes, 3]
                
                # Calculate errors for each prediction step
                pred_cpu = pred_standard.cpu()
                target_cpu = target_batch
                
                for step in range(min(self.config.prediction_horizon, pred_cpu.shape[1])):
                    if step < target_cpu.shape[0]:
                        step_pred = pred_cpu[0, step] if pred_cpu.dim() == 4 else pred_cpu[0]
                        step_target = target_cpu[step]
                        
                        # Calculate RMSE for this step
                        mse = torch.mean((step_pred - step_target) ** 2)
                        rmse = torch.sqrt(mse).item()
                        step_errors[f'step_{step+1}'].append(rmse)
                
                all_predictions.append(pred_cpu)
                all_targets.append(target_cpu)
                
            except Exception as e:
                print(f"    ⚠️ Error in sequence {seq_idx}: {e}")
                continue
        
        # Calculate metrics
        metrics = {}
        for step, errors in step_errors.items():
            if errors:
                metrics[f'{step}_rmse'] = np.mean(errors)
                metrics[f'{step}_std'] = np.std(errors)
        
        # Overall metrics
        all_step_errors = [err for errors in step_errors.values() for err in errors]
        if all_step_errors:
            metrics['overall_rmse'] = np.mean(all_step_errors)
            metrics['overall_std'] = np.std(all_step_errors)
            metrics['num_sequences'] = len([errors for errors in step_errors.values() if errors])
        
        print(f"  ✅ {model_wrapper.config.name}: Overall RMSE = {metrics.get('overall_rmse', 0):.3f}")
        
        return metrics
    
    def run_comparison(self, model_configs: List[ModelConfig]) -> pd.DataFrame:
        """Run comparison across all models"""
        
        print("🏁 Starting unified model comparison...")
        
        # Setup test data
        self.setup_test_data()
        
        # Evaluate each model
        comparison_results = []
        
        for config in model_configs:
            wrapper = ModelWrapper(config, self.device)
            metrics = self.evaluate_model(wrapper)
            
            result_row = {
                'model_name': config.name,
                'model_path': config.model_path,
                **metrics
            }
            comparison_results.append(result_row)
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(comparison_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"model_comparison_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        print(f"📊 Comparison complete! Results saved to: {results_path}")
        
        return results_df
    
    def create_comparison_plots(self, results_df: pd.DataFrame):
        """Create visualization of model comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall RMSE comparison
        if 'overall_rmse' in results_df.columns:
            axes[0, 0].bar(results_df['model_name'], results_df['overall_rmse'])
            axes[0, 0].set_title('Overall RMSE Comparison')
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Step-by-step comparison
        step_cols = [col for col in results_df.columns if 'step_' in col and '_rmse' in col]
        if step_cols:
            for idx, model in results_df.iterrows():
                step_rmses = [model[col] for col in step_cols if pd.notna(model[col])]
                axes[0, 1].plot(range(1, len(step_rmses)+1), step_rmses, 
                               marker='o', label=model['model_name'])
            axes[0, 1].set_title('RMSE by Prediction Step')
            axes[0, 1].set_xlabel('Prediction Step')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].legend()
        
        # Model parameters vs performance (if available)
        axes[1, 0].text(0.5, 0.5, 'Parameter Count\nvs Performance\n(TODO)', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Summary statistics
        summary_text = "Model Comparison Summary:\n\n"
        for idx, row in results_df.iterrows():
            summary_text += f"{row['model_name']}: {row.get('overall_rmse', 'N/A'):.3f} RMSE\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, ha='left', va='top', 
                        transform=axes[1, 1].transAxes, fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path(f"model_comparison_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Comparison plots saved to: {plot_path}")

def main():
    """Main evaluation function"""
    
    print("🌊 UNIFIED MODEL EVALUATION FRAMEWORK")
    print("=" * 50)
    
    # Define model configurations
    model_configs = [
        # ModelConfig(
        #     name="SpatioTemporal",
        #     model_path="experiments/spatiotemporal_20250702_235533/spatiotemporal_model.pt",
        #     input_features=11,  # UPDATE WITH ACTUAL COUNT
        #     output_format="linear_mwd"
        # ),
        # ModelConfig(
        #     name="SpatioTemporalCircular", 
        #     model_path="experiments/spatiotemporal_circular_20250706_232926/spatiotemporal_circular_model.pt",
        #     input_features=11,  # UPDATE WITH ACTUAL COUNT
        #     output_format="circular_mwd"
        # ),
        ModelConfig(
            name="EnhancedPhysics",
            model_path="experiments/enhanced_physics_20250706_235919/enhanced_physics_model.pt",
            input_features=11,
            output_format="circular_mwd"
        )
        # ModelConfig(
        #     name="ProductionSpatial",
        #     model_path="checkpoints/production_spatial_multiyear_highres_final.pt",
        #     input_features=11,  # UPDATE WITH ACTUAL COUNT
        #     output_format="linear_mwd"
        # )
    ]
    
    # Setup evaluation
    eval_config = EvaluationConfig(
        num_test_sequences=50,  # Configurable test size
        test_start_time=400,
        test_end_time=500
    )
    
    evaluator = UnifiedEvaluator(eval_config)
    
    # Run comparison
    results_df = evaluator.run_comparison(model_configs)
    
    # Display results
    print("\n📊 COMPARISON RESULTS:")
    print(results_df[['model_name', 'overall_rmse', 'step_1_rmse', 'step_4_rmse']].to_string(index=False))
    
    # Create plots
    evaluator.create_comparison_plots(results_df)
    
    print(f"\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()