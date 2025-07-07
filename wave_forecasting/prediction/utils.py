# prediction/utils.py - Model Loading and Setup Utilities

import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from config.base import ExperimentConfig, DataConfig, MeshConfig, ModelConfig
from data.loaders import ERA5DataManager, GEBCODataManager
from data.preprocessing import MultiResolutionInterpolator
from data.datasets import MeshDataLoader
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from models.spatial import SpatialWaveGNN


@dataclass
class PredictionEnvironment:
    """Container for all components needed for prediction"""
    model: torch.nn.Module
    mesh_loader: MeshDataLoader
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    feature_names: List[str]
    config: ExperimentConfig
    experiment_id: str

class ModelLoader:
    """Utility class for loading trained models and recreating prediction environment"""
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path: str, 
                           data_year: int = 2020, 
                           data_month: int = 1) -> PredictionEnvironment:
        """
        Load a complete prediction environment from a model checkpoint
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            data_year, data_month: Year/month for setting up data loaders
        
        Returns:
            PredictionEnvironment with everything needed for prediction
        """
        
        print(f"📦 Loading model from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) # unsafe fix
        
        # Extract configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"✅ Config loaded from checkpoint")
        else:
            # Fallback: create default config
            print("⚠️  No config in checkpoint, using defaults")
            # from config.base import ExperimentConfig, DataConfig, MeshConfig, ModelConfig, TrainingConfig
            config = ExperimentConfig(
                name="loaded_model",
                data=DataConfig(),
                mesh=MeshConfig(refinement_level=5),
                model=ModelConfig(hidden_dim=256, num_spatial_layers=12, edge_features=3, output_features=3),
                training=TrainingConfig()
            )
        
        # Setup data environment (recreate mesh, interpolators, etc.)
        print("🔧 Recreating data environment...")
        prediction_env = ModelLoader._recreate_environment(config, data_year, data_month)
        
        # Load model state
        print("🧠 Loading model state...")
        model_config = ModelConfig(hidden_dim=256, num_spatial_layers=12, edge_features=3, output_features=3)
        model = SpatialWaveGNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Replace model in environment
        prediction_env.model = model
        
        # Extract experiment ID
        experiment_id = checkpoint.get('experiment_id', checkpoint_path.stem)
        prediction_env.experiment_id = experiment_id
        
        print(f"✅ Model loaded successfully!")
        print(f"   Experiment: {experiment_id}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Features: {len(prediction_env.feature_names)}")
        print(f"   Nodes: {len(torch.unique(prediction_env.edge_index))}")
        
        return prediction_env
    
    @staticmethod
    def load_from_experiment_log(log_dir: str, 
                                experiment_id: str,
                                data_year: int = 2020,
                                data_month: int = 1) -> PredictionEnvironment:
        """
        Load model from experiment logging directory
        
        Args:
            log_dir: Base logging directory (e.g., "logs")
            experiment_id: Experiment ID (e.g., "spatial_wave_prediction_v2_20241215_143022")
        """
        
        log_path = Path(log_dir) / experiment_id
        
        # Look for final model or latest checkpoint
        model_paths = [
            log_path / "final_model.pt",
            log_path / "checkpoints" / "final_model.pt"
        ]
        
        # Find checkpoint files
        if (log_path / "checkpoints").exists():
            checkpoint_files = list((log_path / "checkpoints").glob("*.pt"))
            model_paths.extend(sorted(checkpoint_files, reverse=True))  # Latest first
        
        # Try loading from available paths
        for model_path in model_paths:
            if model_path.exists():
                print(f"📁 Found model at: {model_path}")
                return ModelLoader.load_from_checkpoint(str(model_path), data_year, data_month)
        
        raise FileNotFoundError(f"No model found for experiment {experiment_id} in {log_dir}")
    
    @staticmethod
    def _recreate_environment(config: ExperimentConfig, 
                            data_year: int, 
                            data_month: int) -> PredictionEnvironment:
        """Recreate the data environment (mesh, loaders, etc.)"""
        
        # Data managers
        data_config = DataConfig()
        era5_manager = ERA5DataManager(data_config)
        gebco_manager = GEBCODataManager(data_config)
        
        # Load sample data for setup
        era5_atmo, era5_waves = era5_manager.load_month_data(data_year, data_month)
        gebco_data = gebco_manager.load_bathymetry()
        
        # Create mesh
        mesh_config = MeshConfig()
        mesh = IcosahedralMesh(mesh_config)
        
        # Create interpolator and mesh loader
        interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, gebco_data, data_config)
        mesh_loader = MeshDataLoader(mesh, interpolator, data_config)
        
        # Get feature names from sample data
        sample_features = mesh_loader.load_features(time_idx=0)
        feature_names = sample_features['feature_names']
        
        # Create graph connectivity
        region_indices = mesh.filter_region(data_config.lat_bounds, data_config.lon_bounds)
        edge_index, edge_attr = compute_regional_edges(mesh, region_indices, mesh_config.max_edge_distance_km)
        
        # Create placeholder environment (model will be replaced)
        model_config = ModelConfig()
        dummy_model = SpatialWaveGNN(model_config)
        
        return PredictionEnvironment(
            model=dummy_model,
            mesh_loader=mesh_loader,
            edge_index=edge_index,
            edge_attr=edge_attr,
            feature_names=feature_names,
            config=config,
            experiment_id="loading"
        )

def list_available_models(log_dir: str = "logs") -> List[Dict[str, Any]]:
    """List all available trained models"""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return []
    
    models = []
    
    for exp_dir in log_path.iterdir():
        if exp_dir.is_dir():
            # Check for model files
            model_files = []
            
            # Check for final models
            if (exp_dir / "final_model.pt").exists():
                model_files.append("final_model.pt")
            
            # Check for checkpoints
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                model_files.extend([f"checkpoints/{cp.name}" for cp in checkpoints])
            
            # Check for config/results
            config_file = exp_dir / "config.json"
            results_exist = config_file.exists()
            
            if model_files:
                models.append({
                    'experiment_id': exp_dir.name,
                    'path': str(exp_dir),
                    'model_files': model_files,
                    'has_config': results_exist,
                    'latest_checkpoint': model_files[0] if model_files else None
                })
    
    print(f"📋 Found {len(models)} trained models:")
    for model in models:
        print(f"   🔹 {model['experiment_id']}")
        print(f"      Files: {', '.join(model['model_files'][:3])}...")
    
    return models

def setup_prediction_environment(experiment_id: str = None, 
                                checkpoint_path: str = None,
                                log_dir: str = "logs") -> PredictionEnvironment:
    """
    Convenient function to setup prediction environment
    
    Args:
        experiment_id: Load from experiment logs  
        checkpoint_path: Load from specific checkpoint
        log_dir: Base log directory
    
    Returns:
        Ready-to-use prediction environment
    """
    
    if checkpoint_path:
        return ModelLoader.load_from_checkpoint(checkpoint_path)
    elif experiment_id:
        return ModelLoader.load_from_experiment_log(log_dir, experiment_id)
    else:
        # Auto-detect latest model
        models = list_available_models(log_dir)
        if not models:
            raise FileNotFoundError("No trained models found!")
        
        latest_model = sorted(models, key=lambda x: x['experiment_id'])[-1]
        print(f"🔄 Auto-loading latest model: {latest_model['experiment_id']}")
        
        return ModelLoader.load_from_experiment_log(log_dir, latest_model['experiment_id'])
