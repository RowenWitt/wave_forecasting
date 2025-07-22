# reconstruct_clean_dataset.py
"""
Reconstruct clean dataset from CDS timestamp duplication issues
Takes the corrupted file with 3x duplicate timestamps and creates a clean version
with proper temporal axis (~120 timesteps instead of 360)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import json

def check_and_handle_expver(file_path: str):
    """Check for problematic expver dimension and provide handling strategy"""
    
    print(f"🔍 CHECKING EXPVER DIMENSION")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Check if expver exists
            has_expver_coord = 'expver' in ds.coords
            has_expver_dim = any('expver' in var.dims for var in ds.data_vars.values())
            
            if has_expver_coord:
                expver_values = ds.expver.values
                unique_expver = np.unique(expver_values)
                print(f"   📊 expver coordinate found:")
                print(f"      Values: {expver_values[:10]}{'...' if len(expver_values) > 10 else ''}")
                print(f"      Unique values: {unique_expver}")
                print(f"      Length: {len(expver_values)}")
                
                if len(unique_expver) == 1:
                    print(f"   ✅ Single expver value ({unique_expver[0]}) - safe to drop")
                else:
                    print(f"   ⚠️  Multiple expver values - need careful handling")
            
            if has_expver_dim:
                print(f"   📊 Variables with expver dimension:")
                for var_name, var_data in ds.data_vars.items():
                    if 'expver' in var_data.dims:
                        expver_size = var_data.sizes['expver']
                        print(f"      {var_name}: expver size = {expver_size}")
            
            if not has_expver_coord and not has_expver_dim:
                print(f"   ✅ No expver dimension found - no action needed")
                return {'needs_expver_handling': False}
            
            return {
                'needs_expver_handling': True,
                'has_expver_coord': has_expver_coord,
                'has_expver_dim': has_expver_dim,
                'safe_to_drop': len(unique_expver) == 1 if has_expver_coord else True
            }
            
    except Exception as e:
        print(f"   ❌ Error checking expver: {e}")
        return {'needs_expver_handling': False}

def analyze_variable_chunks(file_path: str) -> Dict:
    """Analyze which variables are available in which temporal chunks"""
    
    print(f"🔍 ANALYZING VARIABLE CHUNKS")
    print(f"   File: {Path(file_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # First, check and handle expver dimension
            expver_info = check_and_handle_expver(file_path)
            
            # Get time coordinate
            time_coord = None
            for coord in ds.coords:
                if 'time' in coord.lower():
                    time_coord = coord
                    break
            
            time_values = ds[time_coord].values
            time_pandas = pd.to_datetime(time_values)
            
            print(f"   Total timesteps: {len(time_values)}")
            print(f"   Unique timestamps: {len(pd.unique(time_pandas))}")
            
            # Find the chunk structure
            unique_times = pd.unique(time_pandas)
            chunk_size = len(time_values) // len(unique_times)
            
            print(f"   Detected chunk structure: {len(unique_times)} unique times × {chunk_size} chunks")
            
            # Analyze variable availability by chunk
            chunk_info = {}
            
            for chunk_idx in range(chunk_size):
                start_idx = chunk_idx * len(unique_times)
                end_idx = start_idx + len(unique_times)
                
                print(f"\n   📊 Chunk {chunk_idx} (indices {start_idx}-{end_idx-1}):")
                
                chunk_variables = []
                
                for var_name in ds.data_vars:
                    var_data = ds[var_name]
                    
                    # Handle expver dimension if present
                    if 'expver' in var_data.dims:
                        var_data = var_data.isel(expver=0)  # Take first expver value
                    
                    if len(var_data.dims) >= 2:  # Has spatial dimensions
                        time_dim = var_data.dims[0]
                        
                        # Test a few timesteps in this chunk
                        has_data_count = 0
                        for test_idx in range(start_idx, min(start_idx + 5, end_idx)):
                            time_slice = var_data.isel({time_dim: test_idx})
                            if np.any(np.isfinite(time_slice.values)):
                                has_data_count += 1
                        
                        has_data = has_data_count > 0
                        
                        if has_data:
                            chunk_variables.append(var_name)
                            
                            # Get sample range
                            sample_slice = var_data.isel({time_dim: start_idx})
                            sample_values = sample_slice.values
                            finite_values = sample_values[np.isfinite(sample_values)]
                            
                            if len(finite_values) > 0:
                                sample_range = f"[{finite_values.min():.2f}, {finite_values.max():.2f}]"
                            else:
                                sample_range = "All NaN"
                            
                            print(f"      ✅ {var_name}: {sample_range}")
                
                chunk_info[chunk_idx] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'variables': chunk_variables,
                    'time_indices': list(range(start_idx, end_idx))
                }
                
                print(f"      Total variables with data: {len(chunk_variables)}")
            
            return {
                'chunk_info': chunk_info,
                'unique_times': unique_times,
                'time_coord': time_coord,
                'chunk_size': chunk_size,
                'total_timesteps': len(time_values),
                'expver_info': expver_info
            }
            
    except Exception as e:
        print(f"❌ Error analyzing chunks: {e}")
        import traceback
        traceback.print_exc()
        return None

def reconstruct_clean_dataset(file_path: str, output_path: str, chunk_analysis: Dict) -> bool:
    """Reconstruct a clean dataset by selecting the best data from each chunk"""
    
    print(f"\n🔧 RECONSTRUCTING CLEAN DATASET")
    print(f"   Input: {Path(file_path).name}")
    print(f"   Output: {Path(output_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            chunk_info = chunk_analysis['chunk_info']
            unique_times = chunk_analysis['unique_times']
            time_coord = chunk_analysis['time_coord']
            
            print(f"   Reconstructing {len(unique_times)} clean timesteps from {chunk_analysis['total_timesteps']} corrupted ones")
            
            # Create mapping of variables to their best chunks
            variable_to_chunk = {}
            
            for chunk_idx, info in chunk_info.items():
                for var_name in info['variables']:
                    if var_name not in variable_to_chunk:
                        variable_to_chunk[var_name] = chunk_idx
                        print(f"   📍 {var_name} -> Chunk {chunk_idx}")
                    else:
                        print(f"   ⚠️  {var_name} available in multiple chunks (using chunk {variable_to_chunk[var_name]})")
            
            print(f"\n   📊 Variable distribution across chunks:")
            for chunk_idx, info in chunk_info.items():
                assigned_vars = [var for var, chunk in variable_to_chunk.items() if chunk == chunk_idx]
                print(f"      Chunk {chunk_idx}: {len(assigned_vars)} variables")
            
            # Create clean dataset by combining the best data from each chunk
            clean_data_vars = {}
            clean_coords = {}
            
            # Use the first chunk's time indices as the target (they should all be the same unique times)
            target_chunk = 0  # Use chunk 0 as reference
            target_indices = list(range(len(unique_times)))
            
            print(f"\n   🔗 Combining data:")
            
            for var_name, best_chunk in variable_to_chunk.items():
                var_data = ds[var_name]
                
                # Drop expver dimension if present
                if 'expver' in var_data.dims:
                    print(f"      🗑️  Dropping expver from {var_name}")
                    # Take first expver value (they should all be the same for <2023 data)
                    var_data = var_data.isel(expver=0)
                
                # Get the indices for the best chunk
                chunk_start = chunk_info[best_chunk]['start_idx']
                chunk_indices = [chunk_start + i for i in target_indices]
                
                # Extract clean data
                clean_var_data = var_data.isel({time_coord: chunk_indices})
                clean_data_vars[var_name] = clean_var_data
                
                print(f"      ✅ {var_name}: extracted from chunk {best_chunk}")
            
            # Handle coordinates
            for coord_name, coord_data in ds.coords.items():
                if coord_name == time_coord:
                    # Use clean time coordinate (from first chunk)
                    clean_time_indices = [chunk_info[0]['start_idx'] + i for i in target_indices]
                    clean_coords[coord_name] = ds[coord_name].isel({coord_name: clean_time_indices})
                elif coord_name.lower() == 'expver':
                    # Drop expver coordinate - it's problematic and unnecessary for <2023 data
                    print(f"      🗑️  Dropping problematic expver coordinate")
                    continue
                else:
                    # Copy other coordinates as-is
                    clean_coords[coord_name] = coord_data
            
            # Create clean dataset
            clean_ds = xr.Dataset(clean_data_vars, coords=clean_coords)
            
            # Add metadata - CONVERT BOOLEAN TO STRING TO AVOID NetCDF4 TYPE ERROR
            clean_ds.attrs.update(ds.attrs)
            clean_ds.attrs.update({
                'title': 'Reconstructed ERA5 single-level data (CDS duplication fixed)',
                'source': 'ERA5 reanalysis - reconstructed from corrupted CDS download',
                'reconstruction_date': pd.Timestamp.now().isoformat(),
                'original_file': Path(file_path).name,
                'original_timesteps': str(chunk_analysis['total_timesteps']),  # Convert to string
                'reconstructed_timesteps': str(len(unique_times)),  # Convert to string
                'chunk_reconstruction': 'true',  # Convert boolean to string
                'variables_per_chunk': json.dumps({f'chunk_{k}': len(v['variables']) for k, v in chunk_info.items()})
            })
            
            # Verify reconstruction
            print(f"\n   ✅ Reconstruction verification:")
            print(f"      Original timesteps: {chunk_analysis['total_timesteps']}")
            print(f"      Clean timesteps: {len(clean_ds[time_coord])}")
            print(f"      Variables reconstructed: {len(clean_data_vars)}")
            
            # Check time coordinate
            clean_times = pd.to_datetime(clean_ds[time_coord].values)
            print(f"      Time range: {clean_times[0]} to {clean_times[-1]}")
            print(f"      No duplicate timestamps: {len(clean_times) == len(pd.unique(clean_times))}")
            
            # Quick data quality check
            print(f"\n   🔬 Data quality check:")
            for var_name in list(clean_data_vars.keys())[:5]:  # Check first 5 variables
                var_data = clean_ds[var_name]
                if len(var_data.dims) >= 2:
                    values = var_data.values
                    finite_percent = (np.isfinite(values).sum() / values.size) * 100
                    print(f"      {var_name}: {finite_percent:.1f}% finite data")
            
            # Save clean dataset
            print(f"\n   💾 Saving clean dataset...")
            os.makedirs(Path(output_path).parent, exist_ok=True)
            
            # Use compression
            encoding = {}
            for var in clean_data_vars:
                encoding[var] = {'zlib': True, 'complevel': 1}
            
            clean_ds.to_netcdf(output_path, encoding=encoding)
            
            # Final verification
            print(f"   ✅ Verifying saved file...")
            with xr.open_dataset(output_path) as verify_ds:
                print(f"      File size: {Path(output_path).stat().st_size / (1024**2):.1f} MB")
                print(f"      Variables: {len(verify_ds.data_vars)}")
                print(f"      Time steps: {len(verify_ds[time_coord])}")
                print(f"      Dimensions: {dict(verify_ds.dims)}")
            
            print(f"   🎉 Clean dataset successfully created!")
            return True
            
    except Exception as e:
        print(f"❌ Error reconstructing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_reconstruct_all_files(directory: str):
    """Reconstruct all corrupted single-level files in a directory"""
    
    print(f"🔄 BATCH RECONSTRUCTION")
    print(f"   Directory: {directory}")
    print("=" * 60)
    
    directory = Path(directory)
    
    # Find all single-level files
    single_files = list(directory.glob("era5_single_level_*.nc"))
    
    if not single_files:
        print(f"❌ No single-level files found")
        return {}
    
    print(f"📁 Found {len(single_files)} single-level files")
    
    results = {}
    
    for single_file in single_files:
        
        # Extract time key
        import re
        time_match = re.search(r'era5_single_level_(\d{6})\.nc', single_file.name)
        if not time_match:
            print(f"⚠️  Could not parse time from {single_file.name}")
            continue
        
        time_key = time_match.group(1)
        output_file = directory / f"era5_single_level_{time_key}_CLEAN.nc"
        
        print(f"\n📅 Processing {time_key}:")
        
        # Check if output already exists
        if output_file.exists():
            print(f"   ⚠️  Clean file already exists: {output_file.name}")
            response = input(f"   Overwrite? (y/N): ").lower().strip()
            if response != 'y':
                results[time_key] = False
                continue
        
        # Analyze chunks
        chunk_analysis = analyze_variable_chunks(str(single_file))
        
        if chunk_analysis is None:
            print(f"   ❌ Failed to analyze chunks")
            results[time_key] = False
            continue
        
        # Check if reconstruction is needed
        if chunk_analysis['chunk_size'] <= 1:
            print(f"   ✅ File appears clean (no duplication)")
            results[time_key] = True
            continue
        
        # Reconstruct
        success = reconstruct_clean_dataset(str(single_file), str(output_file), chunk_analysis)
        results[time_key] = success
        
        if success:
            print(f"   ✅ Successfully reconstructed {time_key}")
        else:
            print(f"   ❌ Failed to reconstruct {time_key}")
    
    # Summary
    print(f"\n📊 BATCH RECONSTRUCTION SUMMARY:")
    successful = sum(results.values())
    total = len(results)
    print(f"   Processed: {total} files")
    print(f"   Successful: {successful}")
    print(f"   Failed: {total - successful}")
    
    if successful > 0:
        print(f"\n✅ Clean files created:")
        for time_key, success in results.items():
            if success:
                print(f"   era5_single_level_{time_key}_CLEAN.nc")
    
    return results

def main():
    """Main reconstruction function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconstruct clean ERA5 datasets from CDS duplication')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output', help='Output file (for single file mode)')
    parser.add_argument('--batch', action='store_true', help='Process all files in directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input not found: {input_path}")
        return
    
    if args.batch or input_path.is_dir():
        # Batch mode
        batch_reconstruct_all_files(str(input_path))
    else:
        # Single file mode
        if not args.output:
            # Generate output filename
            stem = input_path.stem
            output_path = input_path.parent / f"{stem}_CLEAN.nc"
        else:
            output_path = Path(args.output)
        
        print(f"🔧 SINGLE FILE RECONSTRUCTION")
        print(f"   Input: {input_path.name}")
        print(f"   Output: {output_path.name}")
        
        # Analyze and reconstruct
        chunk_analysis = analyze_variable_chunks(str(input_path))
        
        if chunk_analysis:
            success = reconstruct_clean_dataset(str(input_path), str(output_path), chunk_analysis)
            
            if success:
                print(f"\n🎉 SUCCESS! Clean dataset created: {output_path.name}")
                print(f"   You can now use this file for training!")
            else:
                print(f"\n❌ FAILED to reconstruct dataset")
        else:
            print(f"\n❌ FAILED to analyze input file")

if __name__ == "__main__":
    main()