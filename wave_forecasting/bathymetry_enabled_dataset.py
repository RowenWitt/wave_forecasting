#!/usr/bin/env python3
"""
Bathymetry-Enabled HybridSamplingDataset
Direct implementation without decorators to avoid pickling issues
"""

from typing import List, Dict, Optional
import numpy as np
import torch
from pathlib import Path

# Import the original HybridSamplingDataset
from global_wave_model_hybrid_sampling_v3 import HybridSamplingDataset

# Import the bathymetry handler
from bathymetry_model_dual_purpose import DualPurposeBathymetry


class BathymetryEnabledHybridSamplingDataset(HybridSamplingDataset):
    """
    HybridSamplingDataset with integrated bathymetry support
    Extends the base class to add ocean masking and proper depth features
    """
    
    def __init__(self, data_paths: List[str], mesh, config, gebco_path: str = None):
        """
        Initialize dataset with optional bathymetry support
        
        Args:
            data_paths: List of data file paths
            mesh: Icosahedral mesh object
            config: HybridSamplingConfig
            gebco_path: Path to GEBCO bathymetry file (optional)
        """
        # Initialize parent class first
        super().__init__(data_paths, mesh, config)
        
        # Add bathymetry support if path provided
        if gebco_path:
            print("\n🌊 Initializing bathymetry support...")
            
            # Create dual-purpose bathymetry handler
            self.bathymetry = DualPurposeBathymetry(
                gebco_path=gebco_path,
                cache_dir=f"cache/bathymetry_dual/{Path(config.output_dir).name}",
                resolution=0.1,
                ocean_threshold=-10.0,
                max_depth=5000.0,
                normalize_depth=True
            )
            
            # Get bathymetry data for mesh
            bath_data = self.bathymetry.get_mesh_bathymetry(self.mesh)
            self.ocean_mask = bath_data['mask']
            self.mesh_bathymetry = bath_data['normalized_depth']
            
            print(f"🌊 Bathymetry integrated: mask + feature")
            print(f"   Ocean nodes: {self.ocean_mask.sum():,}/{len(self.ocean_mask):,}")
        else:
            print("⚠️  No GEBCO path provided - using data-based ocean detection")
            self.ocean_mask = None
            self.mesh_bathymetry = None
            self.bathymetry = None
    
    def _interpolate_to_mesh(self, field_data: np.ndarray, 
                            variable_name: Optional[str] = None,
                            is_ocean_variable: bool = False) -> np.ndarray:
        """
        Enhanced interpolation with ocean mask support
        
        Args:
            field_data: Regular grid data to interpolate
            variable_name: Name of the variable (for special handling)
            is_ocean_variable: Whether this is an ocean-only variable
            
        Returns:
            Interpolated data on mesh nodes
        """
        # Special handling for ocean_depth - use pre-computed bathymetry
        if variable_name == 'ocean_depth' and self.mesh_bathymetry is not None:
            return self.mesh_bathymetry
        
        # Regular interpolation using parent method
        if hasattr(super(), '_interpolate_to_mesh'):
            # Check if parent expects is_ocean_variable parameter
            import inspect
            parent_sig = inspect.signature(super()._interpolate_to_mesh)
            if 'is_ocean_variable' in parent_sig.parameters:
                interpolated = super()._interpolate_to_mesh(field_data, is_ocean_variable=is_ocean_variable)
            else:
                interpolated = super()._interpolate_to_mesh(field_data)
        else:
            # Fallback if parent doesn't have the method
            raise NotImplementedError("Parent class must implement _interpolate_to_mesh")
        
        # Apply ocean mask for ocean variables
        if self.ocean_mask is not None and (is_ocean_variable or 
            variable_name in ['swh', 'mwd', 'mwp', 'shww', 'sst']):
            # Zero out land nodes
            interpolated[~self.ocean_mask] = 0.0
        
        return interpolated
    
    def __getitem__(self, idx):
        """
        Get a sample with enhanced bathymetry support
        """
        # Get base sample from parent
        sample = super().__getitem__(idx)
        
        # If we have bathymetry, replace the ocean_depth feature
        if self.mesh_bathymetry is not None and 'ocean_depth' in self.config.input_features:
            depth_idx = self.config.input_features.index('ocean_depth')
            
            # Replace with proper bathymetry for all timesteps
            for t in range(sample['input'].shape[0]):
                sample['input'][t, :, depth_idx] = torch.tensor(
                    self.mesh_bathymetry, 
                    dtype=torch.float32
                )
        
        return sample
    
    def get_ocean_statistics(self) -> Dict[str, float]:
        """
        Get statistics about ocean coverage
        
        Returns:
            Dictionary with ocean statistics
        """
        if self.ocean_mask is None:
            return {"ocean_percentage": 0.0, "ocean_nodes": 0, "total_nodes": 0}
        
        ocean_nodes = int(self.ocean_mask.sum())
        total_nodes = len(self.ocean_mask)
        ocean_percentage = ocean_nodes / total_nodes * 100
        
        return {
            "ocean_percentage": ocean_percentage,
            "ocean_nodes": ocean_nodes,
            "total_nodes": total_nodes,
            "mean_depth": float(self.mesh_bathymetry[self.ocean_mask].mean()) if ocean_nodes > 0 else 0.0,
            "max_depth": float(self.mesh_bathymetry.max())
        }


# For backward compatibility - direct usage without decorator
def create_bathymetry_enabled_dataset(data_paths: List[str], mesh, config, gebco_path: str = None):
    """
    Factory function to create bathymetry-enabled dataset
    
    This replaces the decorator pattern with a simple factory function
    """
    return BathymetryEnabledHybridSamplingDataset(
        data_paths=data_paths,
        mesh=mesh,
        config=config,
        gebco_path=gebco_path
    )