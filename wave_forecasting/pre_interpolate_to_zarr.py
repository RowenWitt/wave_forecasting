#!/usr/bin/env python3
"""
Pre-interpolate ERA5 data to icosahedral mesh and save as Zarr
This creates efficient pre-interpolated datasets for fast training
"""

import numpy as np
import xarray as xr
from pathlib import Path
import glob
import time
from typing import List, Dict, Optional
import hashlib
import json
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

# Import your existing mesh and config
from global_wave_model_hybrid_sampling_v3 import (
    MultiscaleGlobalIcosahedralMesh, 
    HybridSamplingConfig,
    DualPurposeBathymetry
)


class MeshInterpolator:
    """Pre-interpolate ERA5 data to icosahedral mesh"""
    
    def __init__(self, 
                 mesh,
                 config: HybridSamplingConfig,
                 output_dir: str = "data/v1_global/interpolated",
                 gebco_path: Optional[str] = None):
        
        self.mesh = mesh
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get mesh coordinates
        self.mesh_lats, self.mesh_lons = mesh.vertices_to_lat_lon()
        self.n_nodes = len(self.mesh_lats)
        self.mesh_points = np.column_stack([self.mesh_lats, self.mesh_lons])
        
        # Setup bathymetry if provided
        self.bathymetry = None
        self.ocean_mask = None
        if gebco_path:
            print("🌊 Setting up bathymetry...")
            self.bathymetry = DualPurposeBathymetry(
                gebco_path=gebco_path,
                resolution=0.1,
                ocean_threshold=-10.0
            )
            bath_data = self.bathymetry.get_mesh_bathymetry(self.mesh)
            self.ocean_mask = bath_data['mask']
            self.ocean_depth = bath_data['normalized_depth']
        
        # Variables to interpolate
        self.variables = self.config.input_features + self.config.target_features
        # Remove duplicates while preserving order
        self.variables = list(dict.fromkeys(self.variables))
        
        print(f"🔧 Mesh Interpolator initialized:")
        print(f"   Mesh nodes: {self.n_nodes:,}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Variables to interpolate: {len(self.variables)}")
        if self.ocean_mask is not None:
            print(f"   Ocean nodes: {self.ocean_mask.sum():,} ({self.ocean_mask.sum()/self.n_nodes*100:.1f}%)")
    
    def _get_output_path(self, input_path: str) -> Path:
        """Generate output path for interpolated file"""
        input_name = Path(input_path).stem
        mesh_hash = hashlib.md5(f"{self.n_nodes}_{self.config.mesh_refinement_level}".encode()).hexdigest()[:8]
        output_name = f"{input_name}_mesh{self.n_nodes}_{mesh_hash}.zarr"
        return self.output_dir / output_name
    
    def _create_metadata(self, input_path: str, n_timesteps: int, variables_found: List[str]) -> Dict:
        """Create metadata for interpolated file"""
        return {
            'source_file': str(input_path),
            'mesh_nodes': self.n_nodes,
            'mesh_refinement_level': self.config.mesh_refinement_level,
            'n_timesteps': n_timesteps,
            'variables': variables_found,
            'interpolation_method': 'linear',
            'created_timestamp': time.time(),
            'config': {
                'sequence_length': self.config.sequence_length,
                'lat_bounds': self.config.lat_bounds,
                'lon_bounds': self.config.lon_bounds
            }
        }
    
    def interpolate_file(self, input_path: str, force: bool = False) -> Path:
        """
        Interpolate a single ERA5 file to mesh
        
        Args:
            input_path: Path to ERA5 netCDF file
            force: Force re-interpolation even if output exists
            
        Returns:
            Path to output zarr file
        """
        output_path = self._get_output_path(input_path)
        
        # Check if already exists
        if output_path.exists() and not force:
            print(f"✅ Already interpolated: {output_path.name}")
            return output_path
        
        print(f"\n🔄 Interpolating {Path(input_path).name}...")
        start_time = time.time()
        
        # Open source dataset
        with xr.open_dataset(input_path) as ds:
            # Determine time dimension
            time_dim = 'time' if 'time' in ds.dims else 'valid_time'
            n_timesteps = len(ds[time_dim])
            
            print(f"   Source grid: {len(ds.latitude)} x {len(ds.longitude)}")
            print(f"   Timesteps: {n_timesteps}")
            print(f"   Target mesh: {self.n_nodes:,} nodes")
            
            # Create coordinates for output
            coords = {
                time_dim: ds[time_dim].values,
                'node': np.arange(self.n_nodes),
                'lat': ('node', self.mesh_lats),
                'lon': ('node', self.mesh_lons)
            }
            
            # Add ocean mask as coordinate if available
            if self.ocean_mask is not None:
                coords['ocean_mask'] = ('node', self.ocean_mask)
            
            # Initialize data variables
            data_vars = {}
            variables_found = []
            
            # Get grid coordinates
            lats = ds.latitude.values
            lons = ds.longitude.values
            
            # Process each variable
            for var_idx, var in enumerate(self.variables):
                if var == 'ocean_depth' and self.ocean_depth is not None:
                    # Use pre-computed bathymetry
                    print(f"   [{var_idx+1}/{len(self.variables)}] Using pre-computed ocean_depth")
                    # Repeat for all timesteps
                    depth_data = np.tile(self.ocean_depth[np.newaxis, :], (n_timesteps, 1))
                    data_vars['ocean_depth'] = ([time_dim, 'node'], depth_data.astype(np.float32))
                    variables_found.append('ocean_depth')
                    continue
                
                if var not in ds.variables:
                    print(f"   [{var_idx+1}/{len(self.variables)}] Skipping {var} (not found)")
                    continue
                
                print(f"   [{var_idx+1}/{len(self.variables)}] Interpolating {var}...", end='', flush=True)
                var_start = time.time()
                
                # Check if time-varying
                if time_dim in ds[var].dims:
                    # Time-varying variable
                    output_data = np.zeros((n_timesteps, self.n_nodes), dtype=np.float32)
                    
                    # Process in chunks for memory efficiency
                    chunk_size = 10  # Process 10 timesteps at a time
                    
                    for t_start in range(0, n_timesteps, chunk_size):
                        t_end = min(t_start + chunk_size, n_timesteps)
                        
                        for t in range(t_start, t_end):
                            # Get field data
                            field_data = ds[var].isel(**{time_dim: t}).values
                            
                            # Handle NaN values
                            nan_ratio = np.isnan(field_data).sum() / field_data.size
                            field_data_filled = np.nan_to_num(field_data, nan=0.0)
                            
                            # Create interpolator
                            interpolator = RegularGridInterpolator(
                                (lats, lons), 
                                field_data_filled,
                                method='linear',
                                bounds_error=False,
                                fill_value=0.0
                            )
                            
                            # Interpolate to mesh
                            output_data[t, :] = interpolator(self.mesh_points)
                            
                            # Apply ocean mask for ocean variables
                            if self.ocean_mask is not None and var in ['swh', 'mwd', 'mwp', 'shww', 'sst']:
                                output_data[t, ~self.ocean_mask] = 0.0
                        
                        # Progress indicator
                        if t_end % 50 == 0:
                            print('.', end='', flush=True)
                    
                    data_vars[var] = ([time_dim, 'node'], output_data)
                    variables_found.append(var)
                    
                else:
                    # Static variable (shouldn't happen for wave/atmospheric data)
                    print(" (static)", end='')
                    field_data = ds[var].values
                    field_data_filled = np.nan_to_num(field_data, nan=0.0)
                    
                    interpolator = RegularGridInterpolator(
                        (lats, lons), 
                        field_data_filled,
                        method='linear',
                        bounds_error=False,
                        fill_value=0.0
                    )
                    
                    static_data = interpolator(self.mesh_points).astype(np.float32)
                    
                    # Repeat for all timesteps
                    output_data = np.tile(static_data[np.newaxis, :], (n_timesteps, 1))
                    data_vars[var] = ([time_dim, 'node'], output_data)
                    variables_found.append(var)
                
                print(f" ({time.time() - var_start:.1f}s)")
            
            # Create output dataset
            print(f"\n   Creating output dataset...")
            out_ds = xr.Dataset(data_vars=data_vars, coords=coords)
            
            # Add attributes
            out_ds.attrs.update(self._create_metadata(input_path, n_timesteps, variables_found))
            
            # Save as zarr with compression
            print(f"   Saving to {output_path}...")
            
            # Configure encoding for compression
            encoding = {}
            for var in data_vars:
                encoding[var] = {
                    'chunks': (10, min(10000, self.n_nodes))  # Chunk by time and nodes
                    # Let xarray/zarr handle compression automatically
                }
            
            # Save
            out_ds.to_zarr(output_path, mode='w', encoding=encoding)
            
            # Save metadata as separate JSON for quick access
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(out_ds.attrs, f, indent=2)
        
        elapsed = time.time() - start_time
        file_size_mb = sum(f.stat().st_size for f in output_path.rglob('*')) / (1024**2)
        
        print(f"\n✅ Interpolation complete!")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Output size: {file_size_mb:.1f} MB")
        print(f"   Variables: {len(variables_found)}")
        
        return output_path
    
    def interpolate_all(self, 
                       input_pattern: str = "data/v1_global/processed/v1_era5_*.nc",
                       force: bool = False) -> List[Path]:
        """
        Interpolate all files matching pattern
        
        Args:
            input_pattern: Glob pattern for input files
            force: Force re-interpolation
            
        Returns:
            List of output paths
        """
        input_files = sorted(glob.glob(input_pattern))
        
        if not input_files:
            raise ValueError(f"No files found matching pattern: {input_pattern}")
        
        print(f"🌐 Found {len(input_files)} files to interpolate")
        
        output_paths = []
        
        for i, input_path in enumerate(input_files):
            print(f"\n{'='*60}")
            print(f"File {i+1}/{len(input_files)}")
            
            try:
                output_path = self.interpolate_file(input_path, force=force)
                output_paths.append(output_path)
            except Exception as e:
                print(f"❌ Error interpolating {input_path}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"🎉 Interpolation complete!")
        print(f"   Successfully interpolated: {len(output_paths)}/{len(input_files)} files")
        print(f"   Output directory: {self.output_dir}")
        
        # Calculate total size
        total_size_gb = sum(
            sum(f.stat().st_size for f in p.rglob('*')) 
            for p in output_paths
        ) / (1024**3)
        
        print(f"   Total size: {total_size_gb:.1f} GB")
        
        return output_paths


def main():
    """Main function to run pre-interpolation"""
    
    print("🌊 ERA5 to Mesh Pre-interpolation Tool")
    print("=" * 60)
    
    # Configuration
    config = HybridSamplingConfig(
        mesh_refinement_level=6,  # ~40k nodes
        cache_dir="cache/global_mesh_hybrid"
    )
    
    # Create mesh
    print("\n📐 Creating icosahedral mesh...")
    mesh = MultiscaleGlobalIcosahedralMesh(
        refinement_level=config.mesh_refinement_level,
        config=config,
        cache_dir=config.cache_dir
    )
    
    # Create interpolator
    interpolator = MeshInterpolator(
        mesh=mesh,
        config=config,
        output_dir="data/v1_global/interpolated",
        gebco_path="data/gebco/GEBCO_2023.nc"  # Optional, for bathymetry
    )
    
    # Interpolate all files
    try:
        output_paths = interpolator.interpolate_all(
            input_pattern="data/v1_global/processed/v1_era5_2021*.nc",  # Start with 2021
            force=False  # Set to True to re-interpolate existing files
        )
        
        print("\n✅ Pre-interpolation complete!")
        print(f"   Ready for fast training with pre-interpolated data")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()