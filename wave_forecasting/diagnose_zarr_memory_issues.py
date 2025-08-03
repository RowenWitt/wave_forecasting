#!/usr/bin/env python3
"""
Diagnostic script to understand zarr file structure
"""

import xarray as xr
import zarr
import json
from pathlib import Path
import numpy as np

def inspect_zarr_files(zarr_dir: str):
    """Inspect zarr files to understand their structure"""
    zarr_dir = Path(zarr_dir)
    zarr_files = sorted(list(zarr_dir.glob("*.zarr")))
    
    if not zarr_files:
        print(f"No zarr files found in {zarr_dir}")
        return
    
    print(f"\n📊 Found {len(zarr_files)} zarr files")
    
    # Inspect first file in detail
    first_file = zarr_files[0]
    print(f"\n🔍 Inspecting: {first_file.name}")
    
    # Open with xarray
    ds = xr.open_zarr(first_file)
    print(f"\n📋 Dataset info:")
    print(ds)
    
    print(f"\n📐 Dimensions:")
    for dim, size in ds.dims.items():
        print(f"  - {dim}: {size}")
    
    print(f"\n📦 Variables:")
    for var in ds.data_vars:
        var_info = ds[var]
        print(f"  - {var}: shape={var_info.shape}, dims={var_info.dims}, dtype={var_info.dtype}")
    
    print(f"\n🔧 Coordinates:")
    for coord in ds.coords:
        coord_info = ds[coord]
        print(f"  - {coord}: shape={coord_info.shape}, dims={coord_info.dims}, dtype={coord_info.dtype}")
        # Show first few values if it's small
        if coord_info.size < 10:
            print(f"    values: {coord_info.values}")
    
    # Check for lat/lon in different possible locations
    print(f"\n🌍 Looking for latitude/longitude data:")
    possible_lat_names = ['latitude', 'lat', 'y', 'nlat', 'node_lat', 'mesh_lat']
    possible_lon_names = ['longitude', 'lon', 'x', 'nlon', 'node_lon', 'mesh_lon']
    
    found_lat = None
    found_lon = None
    
    # Check in variables
    for lat_name in possible_lat_names:
        if lat_name in ds.data_vars:
            found_lat = lat_name
            print(f"  ✓ Found latitude in data_vars: '{lat_name}' with shape {ds[lat_name].shape}")
            break
        elif lat_name in ds.coords:
            found_lat = lat_name
            print(f"  ✓ Found latitude in coords: '{lat_name}' with shape {ds[lat_name].shape}")
            break
    
    for lon_name in possible_lon_names:
        if lon_name in ds.data_vars:
            found_lon = lon_name
            print(f"  ✓ Found longitude in data_vars: '{lon_name}' with shape {ds[lon_name].shape}")
            break
        elif lon_name in ds.coords:
            found_lon = lon_name
            print(f"  ✓ Found longitude in coords: '{lon_name}' with shape {ds[lon_name].shape}")
            break
    
    if not found_lat:
        print(f"  ❌ Latitude not found in standard locations")
    if not found_lon:
        print(f"  ❌ Longitude not found in standard locations")
    
    # Check metadata file
    metadata_path = zarr_dir / f"{first_file.stem}_metadata.json"
    if metadata_path.exists():
        print(f"\n📄 Metadata file found: {metadata_path.name}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Keys: {list(metadata.keys())}")
        if 'lat' in metadata or 'latitude' in metadata:
            print(f"  ✓ Latitude found in metadata")
        if 'lon' in metadata or 'longitude' in metadata:
            print(f"  ✓ Longitude found in metadata")
        if 'mesh_nodes' in metadata:
            print(f"  Mesh nodes: {metadata['mesh_nodes']}")
    
    # Check data shape consistency
    print(f"\n🔍 Checking data shapes for first timestep:")
    time_dim = 'time' if 'time' in ds.dims else 'valid_time'
    first_time = ds[time_dim].isel({time_dim: 0})
    
    for var in ['swh', 'mwd', 'mwp', 'u10', 'v10']:
        if var in ds.data_vars:
            data = ds[var].isel({time_dim: 0})
            print(f"  - {var}: shape={data.shape}, size={data.size}")
    
    # Close dataset
    ds.close()
    
    print(f"\n💡 Recommendations:")
    print("  1. Ensure lat/lon are stored as data variables or coordinates in the zarr file")
    print("  2. All variables should have the same spatial dimensions")
    print("  3. Consider storing lat/lon as 1D arrays indexed by node number")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        zarr_dir = sys.argv[1]
    else:
        zarr_dir = "data/v1_global/interpolated"
    
    inspect_zarr_files(zarr_dir)