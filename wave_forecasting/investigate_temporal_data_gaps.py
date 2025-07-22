# investigate_temporal_data_gaps.py
"""
Investigate temporal data gaps in ERA5 single-level data
Check which timesteps have data vs NaN for each variable
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def analyze_temporal_data_availability(file_path: str):
    """Analyze which timesteps have data for each variable"""
    
    print(f"🕵️ TEMPORAL DATA AVAILABILITY INVESTIGATION")
    print(f"   File: {Path(file_path).name}")
    print("=" * 80)
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Get time coordinate
            time_coord = None
            for coord in ds.coords:
                if 'time' in coord.lower():
                    time_coord = coord
                    break
            
            if not time_coord:
                print(f"❌ No time coordinate found")
                return
            
            time_values = ds[time_coord].values
            time_pandas = pd.to_datetime(time_values)
            
            print(f"📅 Time Analysis:")
            print(f"   Time coordinate: {time_coord}")
            print(f"   Total timesteps: {len(time_values)}")
            print(f"   Time range: {time_pandas[0]} to {time_pandas[-1]}")
            
            # Analyze each variable's temporal availability
            var_availability = {}
            
            print(f"\n📊 Variable Availability by Timestep:")
            print(f"{'Variable':<25} {'Available Steps':<15} {'%':<8} {'First Available':<20} {'Last Available'}")
            print("-" * 100)
            
            for var_name in ds.data_vars:
                var_data = ds[var_name]
                
                if len(var_data.dims) >= 2:  # Has spatial dimensions
                    time_dim = var_data.dims[0]
                    
                    # Check each timestep
                    available_timesteps = []
                    first_available = None
                    last_available = None
                    
                    for t in range(len(time_values)):
                        time_slice = var_data.isel({time_dim: t})
                        has_finite_data = np.any(np.isfinite(time_slice.values))
                        
                        if has_finite_data:
                            available_timesteps.append(t)
                            if first_available is None:
                                first_available = t
                            last_available = t
                    
                    availability_percent = (len(available_timesteps) / len(time_values)) * 100
                    
                    first_time_str = str(time_pandas[first_available])[:19] if first_available is not None else "None"
                    last_time_str = str(time_pandas[last_available])[:19] if last_available is not None else "None"
                    
                    print(f"{var_name:<25} {len(available_timesteps):<15} {availability_percent:<8.1f} {first_time_str:<20} {last_time_str}")
                    
                    var_availability[var_name] = {
                        'available_timesteps': available_timesteps,
                        'availability_percent': availability_percent,
                        'first_available': first_available,
                        'last_available': last_available
                    }
            
            # Analyze patterns
            print(f"\n🔍 PATTERN ANALYSIS:")
            
            # Group variables by availability pattern
            full_availability = []
            partial_availability = []
            no_availability = []
            
            for var_name, info in var_availability.items():
                if info['availability_percent'] > 95:
                    full_availability.append(var_name)
                elif info['availability_percent'] > 5:
                    partial_availability.append(var_name)
                else:
                    no_availability.append(var_name)
            
            print(f"   Variables with FULL data (>95%): {len(full_availability)}")
            if full_availability:
                for var in full_availability:
                    print(f"      ✅ {var}")
            
            print(f"\n   Variables with PARTIAL data (5-95%): {len(partial_availability)}")
            if partial_availability:
                for var in partial_availability:
                    percent = var_availability[var]['availability_percent']
                    print(f"      ⚠️  {var} ({percent:.1f}%)")
            
            print(f"\n   Variables with NO data (<5%): {len(no_availability)}")
            if no_availability:
                for var in no_availability:
                    print(f"      ❌ {var}")
            
            # Check for temporal clustering
            print(f"\n📈 TEMPORAL CLUSTERING ANALYSIS:")
            
            if partial_availability:
                # Analyze one variable with partial data
                test_var = partial_availability[0]
                test_info = var_availability[test_var]
                available_steps = np.array(test_info['available_timesteps'])
                
                if len(available_steps) > 1:
                    # Look for gaps
                    gaps = np.diff(available_steps)
                    large_gaps = gaps[gaps > 1]
                    
                    print(f"   Analyzing {test_var} temporal pattern:")
                    print(f"      Available timesteps: {len(available_steps)}")
                    print(f"      Gaps > 1 timestep: {len(large_gaps)}")
                    
                    if len(large_gaps) > 0:
                        print(f"      Largest gap: {large_gaps.max()} timesteps")
                        print(f"      Average gap: {large_gaps.mean():.1f} timesteps")
                        
                        # Check if data comes in chunks
                        chunk_starts = []
                        chunk_ends = []
                        
                        current_chunk_start = available_steps[0]
                        
                        for i in range(1, len(available_steps)):
                            if available_steps[i] - available_steps[i-1] > 1:
                                # End of chunk
                                chunk_ends.append(available_steps[i-1])
                                chunk_starts.append(available_steps[i])
                        
                        chunk_ends.append(available_steps[-1])
                        
                        print(f"      Data appears in {len(chunk_starts)} chunks:")
                        for start, end in zip([available_steps[0]] + chunk_starts, chunk_ends):
                            start_time = time_pandas[start]
                            end_time = time_pandas[end]
                            chunk_length = end - start + 1
                            print(f"         {start_time} to {end_time} ({chunk_length} steps)")
            
            # Check for specific issues
            print(f"\n🔧 DIAGNOSTIC RECOMMENDATIONS:")
            
            if len(no_availability) > len(full_availability):
                print(f"   🚨 MAJOR ISSUE: More variables have NO data than full data")
                print(f"   🔄 This suggests a serious download problem")
                print(f"   💡 Recommendation: Re-download with different parameters")
            
            elif len(partial_availability) > 0:
                print(f"   ⚠️  Some variables have partial data")
                
                # Check if it's a systematic pattern
                all_available_steps = set()
                for var_name in partial_availability:
                    all_available_steps.update(var_availability[var_name]['available_timesteps'])
                
                common_available_steps = set(var_availability[partial_availability[0]]['available_timesteps'])
                for var_name in partial_availability[1:]:
                    common_available_steps &= set(var_availability[var_name]['available_timesteps'])
                
                if len(common_available_steps) > 0:
                    print(f"   📊 {len(common_available_steps)} timesteps have data for ALL partial variables")
                    print(f"   💡 This suggests systematic temporal gaps, not variable-specific issues")
                else:
                    print(f"   📊 No common timesteps across partial variables")
                    print(f"   💡 This suggests variable-specific download issues")
            
            else:
                print(f"   ✅ All variables have full data availability")
            
            return var_availability
            
    except Exception as e:
        print(f"❌ Error in temporal analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_cds_download_completeness(file_path: str):
    """Check if CDS download seems complete based on file attributes"""
    
    print(f"\n🔍 CDS DOWNLOAD COMPLETENESS CHECK")
    print(f"   File: {Path(file_path).name}")
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            print(f"📋 File Attributes:")
            for attr_name, attr_value in ds.attrs.items():
                print(f"   {attr_name}: {attr_value}")
            
            print(f"\n📊 Dataset Structure:")
            print(f"   Dimensions: {dict(ds.dims)}")
            print(f"   Coordinates: {list(ds.coords.keys())}")
            print(f"   Variables: {len(ds.data_vars)}")
            
            # Check for incomplete download indicators
            print(f"\n🔍 Completeness Indicators:")
            
            # Check file size
            file_size_mb = Path(file_path).stat().st_size / (1024**2)
            expected_size_mb = 50  # Rough estimate for monthly global single-level data
            
            print(f"   File size: {file_size_mb:.1f} MB")
            if file_size_mb < expected_size_mb:
                print(f"   ⚠️  File size seems small (expected ~{expected_size_mb}+ MB)")
            else:
                print(f"   ✅ File size seems reasonable")
            
            # Check for standard ERA5 variables
            expected_vars = ['u10', 'v10', 'msl', 'sst', 'tp', 'swh', 'mwd', 'mwp']
            missing_vars = []
            
            for var in expected_vars:
                found = False
                for actual_var in ds.data_vars:
                    if var in actual_var.lower() or any(var in alias for alias in [
                        '10m_u_component_of_wind', '10m_v_component_of_wind',
                        'mean_sea_level_pressure', 'sea_surface_temperature',
                        'total_precipitation', 'significant_height_of_combined_wind_waves_and_swell',
                        'mean_wave_direction', 'mean_wave_period'
                    ]):
                        found = True
                        break
                
                if not found:
                    missing_vars.append(var)
            
            if missing_vars:
                print(f"   ⚠️  Missing expected variables: {missing_vars}")
            else:
                print(f"   ✅ All expected variables present")
            
            # Check coordinate completeness
            if 'valid_time' in ds.coords:
                time_coord = ds['valid_time']
                print(f"   ✅ Time coordinate present: {len(time_coord)} steps")
            else:
                print(f"   ❌ No valid_time coordinate found")
            
            return {
                'file_size_mb': file_size_mb,
                'missing_vars': missing_vars,
                'has_time_coord': 'valid_time' in ds.coords
            }
            
    except Exception as e:
        print(f"❌ Error checking completeness: {e}")
        return None

def generate_redownload_recommendations(var_availability: Dict, completeness_check: Dict):
    """Generate specific recommendations for re-downloading"""
    
    print(f"\n💡 RE-DOWNLOAD RECOMMENDATIONS")
    print("=" * 50)
    
    # Analyze the issues
    no_data_vars = [var for var, info in var_availability.items() if info['availability_percent'] < 5]
    partial_data_vars = [var for var, info in var_availability.items() if 5 <= info['availability_percent'] <= 95]
    
    if len(no_data_vars) > 0:
        print(f"🚨 CRITICAL ISSUE: {len(no_data_vars)} variables have no data")
        print(f"   Variables: {no_data_vars}")
        print(f"   💡 This suggests the CDS request was incomplete or failed")
        print(f"   🔄 SOLUTION: Re-download with explicit variable list")
        
        print(f"\n📝 Recommended CDS request parameters:")
        print(f"```python")
        print(f"request = {{")
        print(f"    'product_type': 'reanalysis',")
        print(f"    'variable': [")
        print(f"        '10m_u_component_of_wind',")
        print(f"        '10m_v_component_of_wind',")
        print(f"        'mean_sea_level_pressure',")
        print(f"        'sea_surface_temperature',")
        print(f"        'total_precipitation',")
        print(f"        'significant_height_of_combined_wind_waves_and_swell',")
        print(f"        'mean_wave_direction',")
        print(f"        'mean_wave_period'")
        print(f"    ],")
        print(f"    'year': '2019',")
        print(f"    'month': '10',")
        print(f"    'day': [f'{{d:02d}}' for d in range(1, 32)],")
        print(f"    'time': [f'{{h:02d}}:00' for h in range(0, 24, 6)],  # 6-hourly")
        print(f"    'data_format': 'netcdf',")
        print(f"    'area': [90, -180, -90, 180]  # Global")
        print(f"}}```")
    
    elif len(partial_data_vars) > 0:
        print(f"⚠️  PARTIAL DATA ISSUE: {len(partial_data_vars)} variables have gaps")
        print(f"   Variables: {partial_data_vars}")
        
        # Check if gaps are systematic
        first_partial = var_availability[partial_data_vars[0]]
        if first_partial['availability_percent'] > 30:
            print(f"   💡 Data gaps might be due to temporal resolution mismatch")
            print(f"   🔄 SOLUTION: Try downloading at 6-hourly instead of hourly")
        else:
            print(f"   💡 Severe data gaps suggest download issues")
            print(f"   🔄 SOLUTION: Re-download with more conservative parameters")
    
    else:
        print(f"✅ No major download issues detected")
        print(f"   The high NaN percentage is likely due to land vs ocean")

def main():
    """Main investigation function"""
    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python investigate_temporal_data_gaps.py <single_level_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    print(f"🔍 INVESTIGATING TEMPORAL DATA GAPS")
    print(f"=" * 80)
    
    # Run investigations
    var_availability = analyze_temporal_data_availability(file_path)
    
    if var_availability:
        completeness_check = check_cds_download_completeness(file_path)
        
        if completeness_check:
            generate_redownload_recommendations(var_availability, completeness_check)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   If many variables show 0% availability, re-download is needed")
    print(f"   If variables show partial availability, check temporal patterns")
    print(f"   If only wave variables are problematic, they might be ocean-only")

if __name__ == "__main__":
    main()