#!/usr/bin/env python3
"""
Dual-Purpose Bathymetry Handler
Provides both ocean mask and depth feature for wave model
"""

import numpy as np
import xarray as xr
from pathlib import Path
import hashlib
import pickle
import torch
from typing import Tuple, Optional, Dict
from scipy.interpolate import RegularGridInterpolator

class DualPurposeBathymetry:
    """
    Handles bathymetry data for both masking and model features
    Maintains depth values while creating ocean masks
    """
    
    def __init__(self, 
                 gebco_path: str,
                 cache_dir: str = "cache/bathymetry_dual",
                 resolution: float = 0.1,
                 ocean_threshold: float = -10.0,
                 max_depth: float = 5000.0,
                 normalize_depth: bool = True):
        """
        Initialize dual-purpose bathymetry handler
        
        Args:
            gebco_path: Path to GEBCO netCDF file
            cache_dir: Directory for caching processed data
            resolution: Target resolution in degrees (0.1° = ~11km)
            ocean_threshold: Depth threshold for ocean mask (negative = below sea level)
            max_depth: Maximum depth for normalization (meters)
            normalize_depth: Whether to normalize depth values for model
        """
        self.gebco_path = Path(gebco_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.resolution = resolution
        self.ocean_threshold = ocean_threshold
        self.max_depth = max_depth
        self.normalize_depth = normalize_depth
        
        # Create unique cache key
        self.cache_key = self._generate_cache_key()
        
        # Cached data
        self._cached_data = None
        self._mesh_cache = {}  # Cache interpolated values per mesh
        
        print(f"🌊 Dual-Purpose Bathymetry Handler initialized:")
        print(f"   Source: {self.gebco_path.name}")
        print(f"   Resolution: {self.resolution}° (~{self.resolution * 111:.1f} km)")
        print(f"   Ocean threshold: {self.ocean_threshold}m")
        print(f"   Max depth: {self.max_depth}m")
        print(f"   Normalize: {self.normalize_depth}")
    
    def _generate_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        key_parts = [
            f"gebco_{self.gebco_path.stat().st_size}",
            f"res_{self.resolution}",
            f"thresh_{self.ocean_threshold}",
            f"maxd_{self.max_depth}",
            f"norm_{self.normalize_depth}",
            "dual_v1"
        ]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def process_bathymetry(self) -> Dict[str, np.ndarray]:
        """
        Process bathymetry data for dual purpose use
        
        Returns:
            Dictionary with:
                - 'depth': 2D array of depth values (positive = deeper)
                - 'mask': 2D boolean array (True = ocean)
                - 'lats': 1D array of latitudes
                - 'lons': 1D array of longitudes
                - 'normalized_depth': Optional normalized depth for model
        """
        cache_file = self.cache_dir / f"bathymetry_dual_{self.cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            print(f"📂 Loading cached bathymetry data: {cache_file}")
            with open(cache_file, 'rb') as f:
                self._cached_data = pickle.load(f)
            return self._cached_data
        
        print(f"🔨 Processing bathymetry at {self.resolution}° resolution...")
        
        # Define target grid
        target_lats = np.arange(-90, 90 + self.resolution, self.resolution)
        target_lons = np.arange(-180, 180 + self.resolution, self.resolution)
        
        # Initialize arrays
        depth_grid = np.zeros((len(target_lats), len(target_lons)), dtype=np.float32)
        ocean_mask = np.zeros((len(target_lats), len(target_lons)), dtype=bool)
        
        # Process GEBCO data
        with xr.open_dataset(self.gebco_path, chunks={'lat': 5000, 'lon': 5000}) as ds:
            
            # Determine variable name
            if 'elevation' in ds.variables:
                depth_var = 'elevation'
                # For elevation: negative = below sea level, so depth = -elevation
                to_depth = lambda x: -x
                print(f"   Using elevation data (converting to depth)")
            elif 'depth' in ds.variables:
                depth_var = 'depth'
                # For depth: positive = below sea level
                to_depth = lambda x: x
                print(f"   Using depth data directly")
            else:
                raise ValueError("No elevation or depth variable found")
            
            # Process in latitude bands
            lat_bands = np.arange(-90, 91, 10)
            
            for i in range(len(lat_bands) - 1):
                lat_min, lat_max = lat_bands[i], lat_bands[i + 1]
                
                print(f"   Processing {lat_min}° to {lat_max}°...")
                
                # Load band data
                band_data = ds[depth_var].sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(-180, 180)
                ).load()
                
                # Convert to depth (positive = deeper)
                band_depth = to_depth(band_data)
                
                # Find target indices
                lat_idx_start = np.searchsorted(target_lats, lat_min)
                lat_idx_end = np.searchsorted(target_lats, lat_max)
                
                # Downsample to target resolution
                for j, target_lat in enumerate(target_lats[lat_idx_start:lat_idx_end]):
                    for k, target_lon in enumerate(target_lons):
                        # Define window
                        lat_window = (target_lat - self.resolution/2, 
                                     target_lat + self.resolution/2)
                        lon_window = (target_lon - self.resolution/2,
                                     target_lon + self.resolution/2)
                        
                        # Extract window
                        window_data = band_depth.sel(
                            lat=slice(*lat_window),
                            lon=slice(*lon_window)
                        )
                        
                        if window_data.size > 0:
                            # Use mean depth for the cell
                            mean_depth = float(window_data.mean())
                            depth_grid[lat_idx_start + j, k] = mean_depth
                            
                            # Ocean mask: any point in window is ocean
                            ocean_mask[lat_idx_start + j, k] = (window_data > -self.ocean_threshold).any()
        
        # Post-process depth values
        # Set land areas to 0 depth
        depth_grid[~ocean_mask] = 0.0
        
        # Clip extreme depths
        depth_grid = np.clip(depth_grid, 0, self.max_depth)
        
        # Create normalized version for model input
        if self.normalize_depth:
            # Normalize ocean depths to [0, 1] range
            normalized_depth = depth_grid.copy()
            normalized_depth[ocean_mask] = depth_grid[ocean_mask] / self.max_depth
            # Could also use log scaling for better resolution in shallow areas:
            # normalized_depth[ocean_mask] = np.log1p(depth_grid[ocean_mask]) / np.log1p(self.max_depth)
        else:
            normalized_depth = depth_grid
        
        # Calculate statistics
        ocean_cells = ocean_mask.sum()
        total_cells = ocean_mask.size
        mean_ocean_depth = depth_grid[ocean_mask].mean() if ocean_cells > 0 else 0
        
        print(f"✅ Bathymetry processing complete:")
        print(f"   Grid size: {depth_grid.shape}")
        print(f"   Ocean cells: {ocean_cells:,} ({ocean_cells/total_cells*100:.1f}%)")
        print(f"   Mean ocean depth: {mean_ocean_depth:.1f}m")
        print(f"   Max depth in grid: {depth_grid.max():.1f}m")
        
        # Cache the result
        self._cached_data = {
            'depth': depth_grid,
            'mask': ocean_mask,
            'lats': target_lats,
            'lons': target_lons,
            'normalized_depth': normalized_depth,
            'resolution': self.resolution,
            'ocean_threshold': self.ocean_threshold,
            'max_depth': self.max_depth,
            'stats': {
                'ocean_percent': ocean_cells/total_cells*100,
                'mean_ocean_depth': mean_ocean_depth,
                'max_depth': float(depth_grid.max())
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self._cached_data, f)
        
        print(f"💾 Cached bathymetry data: {cache_file}")
        
        return self._cached_data
    
    def get_mesh_bathymetry(self, mesh) -> Dict[str, np.ndarray]:
        """
        Get bathymetry data interpolated to mesh nodes
        
        Args:
            mesh: Icosahedral mesh object
            
        Returns:
            Dictionary with:
                - 'depth': Depth values at mesh nodes (meters)
                - 'mask': Ocean mask at mesh nodes
                - 'normalized_depth': Normalized depth for model input
        """
        # Check if we've already processed this mesh
        mesh_id = id(mesh)
        if mesh_id in self._mesh_cache:
            return self._mesh_cache[mesh_id]
        
        # Ensure we have processed data
        if self._cached_data is None:
            self.process_bathymetry()
        
        # Get mesh coordinates
        mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
        n_nodes = len(mesh_lats)
        
        print(f"\n🎯 Interpolating bathymetry to {n_nodes:,} mesh nodes...")
        
        # Ensure longitude compatibility
        mesh_lons = np.where(mesh_lons > 180, mesh_lons - 360, mesh_lons)
        
        # Create interpolators
        depth_interpolator = RegularGridInterpolator(
            (self._cached_data['lats'], self._cached_data['lons']),
            self._cached_data['depth'],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        mask_interpolator = RegularGridInterpolator(
            (self._cached_data['lats'], self._cached_data['lons']),
            self._cached_data['mask'].astype(float),
            method='nearest',  # Preserve binary nature
            bounds_error=False,
            fill_value=0.0
        )
        
        norm_interpolator = RegularGridInterpolator(
            (self._cached_data['lats'], self._cached_data['lons']),
            self._cached_data['normalized_depth'],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Interpolate
        mesh_coords = np.column_stack([mesh_lats, mesh_lons])
        
        mesh_depth = depth_interpolator(mesh_coords)
        mesh_mask = mask_interpolator(mesh_coords) > 0.5
        mesh_norm_depth = norm_interpolator(mesh_coords)
        
        # Ensure consistency
        mesh_depth[~mesh_mask] = 0.0
        mesh_norm_depth[~mesh_mask] = 0.0
        
        # Statistics
        ocean_nodes = mesh_mask.sum()
        mean_depth = mesh_depth[mesh_mask].mean() if ocean_nodes > 0 else 0
        
        print(f"✅ Mesh bathymetry complete:")
        print(f"   Ocean nodes: {ocean_nodes:,} ({ocean_nodes/n_nodes*100:.1f}%)")
        print(f"   Mean depth at ocean nodes: {mean_depth:.1f}m")
        print(f"   Depth range: [{mesh_depth[mesh_mask].min():.1f}, {mesh_depth[mesh_mask].max():.1f}]m")
        
        # Cache for this mesh
        self._mesh_cache[mesh_id] = {
            'depth': mesh_depth.astype(np.float32),
            'mask': mesh_mask,
            'normalized_depth': mesh_norm_depth.astype(np.float32)
        }
        
        return self._mesh_cache[mesh_id]
    
    def create_mesh_feature(self, mesh) -> np.ndarray:
        """
        Create bathymetry feature for model input
        
        Args:
            mesh: Icosahedral mesh object
            
        Returns:
            Feature array ready for model input
        """
        mesh_data = self.get_mesh_bathymetry(mesh)
        
        # Return normalized depth as feature
        # Land nodes will have 0, ocean nodes will have normalized depth
        return mesh_data['normalized_depth']


def integrate_bathymetry_with_dataset(dataset_class):
    """
    Enhanced decorator to integrate bathymetry as both mask and feature
    """
    class BathymetryEnabledDataset(dataset_class):
        
        def __init__(self, *args, gebco_path: str = None, **kwargs):
            super().__init__(*args, **kwargs)
            
            if gebco_path:
                # Create dual-purpose bathymetry handler
                self.bathymetry = DualPurposeBathymetry(
                    gebco_path=gebco_path,
                    cache_dir=f"cache/bathymetry_dual/{self.config.output_dir.split('/')[-1]}",
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
            else:
                print("⚠️  No GEBCO path provided")
                self.ocean_mask = None
                self.mesh_bathymetry = None
        
        def __getitem__(self, idx):
            """Enhanced getitem that includes proper bathymetry"""
            sample = super().__getitem__(idx)
            
            # If we have bathymetry, replace the broken ocean_depth feature
            if self.mesh_bathymetry is not None:
                # Find ocean_depth index in features
                if 'ocean_depth' in self.config.input_features:
                    depth_idx = self.config.input_features.index('ocean_depth')
                    
                    # Replace with proper bathymetry for all timesteps
                    for t in range(sample['input'].shape[0]):
                        sample['input'][t, :, depth_idx] = torch.tensor(
                            self.mesh_bathymetry, 
                            dtype=torch.float32
                        )
            
            return sample
        
        def _interpolate_to_mesh(self, field_data: np.ndarray, 
                                variable_name: Optional[str] = None,
                                is_ocean_variable: bool = False) -> np.ndarray:
            """Enhanced interpolation with ocean mask"""
            
            # Skip interpolation for ocean_depth - use pre-computed bathymetry
            if variable_name == 'ocean_depth' and self.mesh_bathymetry is not None:
                return self.mesh_bathymetry
            
            # Regular interpolation - call parent with all arguments
            interpolated = super()._interpolate_to_mesh(field_data, is_ocean_variable=is_ocean_variable)
            
            # Apply ocean mask for ocean variables
            if self.ocean_mask is not None and (is_ocean_variable or 
                variable_name in ['swh', 'mwd', 'mwp', 'shww', 'sst']):
                interpolated[~self.ocean_mask] = 0.0
            
            return interpolated
    
    return BathymetryEnabledDataset


# Usage example
if __name__ == "__main__":
    # Test dual-purpose bathymetry
    
    bath_handler = DualPurposeBathymetry(
        gebco_path="data/gebco/GEBCO_2023.nc",
        resolution=0.1,
        ocean_threshold=-10.0,
        max_depth=5000.0,
        normalize_depth=True
    )
    
    # Process bathymetry
    bath_data = bath_handler.process_bathymetry()
    
    print("\n📊 Bathymetry data summary:")
    print(f"   Depth shape: {bath_data['depth'].shape}")
    print(f"   Memory usage: {bath_data['depth'].nbytes / 1024**2:.1f} MB")
    print(f"   Ocean coverage: {bath_data['stats']['ocean_percent']:.1f}%")
    print(f"   Mean ocean depth: {bath_data['stats']['mean_ocean_depth']:.1f}m")
    
    # Test with dummy mesh
    class DummyMesh:
        def vertices_to_lat_lon(self):
            # Sample points including deep ocean, shallow, and land
            lats = np.array([0, 20, 40, -30, 35.6])  # Equator, tropical, mid-lat, southern, land (Tokyo)
            lons = np.array([-140, -150, -130, 30, 139.7])
            return lats, lons
    
    mesh = DummyMesh()
    mesh_bath = bath_handler.get_mesh_bathymetry(mesh)
    
    print("\n🔍 Sample mesh nodes:")
    lats, lons = mesh.vertices_to_lat_lon()
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        print(f"   Node {i}: ({lat:.1f}°, {lon:.1f}°)")
        print(f"      Ocean: {mesh_bath['mask'][i]}")
        print(f"      Depth: {mesh_bath['depth'][i]:.1f}m")
        print(f"      Normalized: {mesh_bath['normalized_depth'][i]:.3f}")
    
    print("\n✅ Dual-purpose bathymetry test complete!")