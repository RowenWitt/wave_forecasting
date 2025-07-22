# analyze_timestamp_duplication.py
"""
Analyze timestamp duplication patterns and variable alignment issues
CDS sometimes creates files with duplicate timestamps where different variables
are available at different instances of the same timestamp
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def analyze_timestamp_duplication(file_path: str):
    """Analyze timestamp duplication and variable availability patterns"""
    
    print(f"🔍 TIMESTAMP DUPLICATION ANALYSIS")
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
                return None
            
            time_values = ds[time_coord].values
            time_pandas = pd.to_datetime(time_values)
            
            print(f"📅 Time Coordinate Analysis:")
            print(f"   Time coordinate: {time_coord}")
            print(f"   Total timesteps: {len(time_values)}")
            print(f"   Unique timestamps: {len(pd.unique(time_pandas))}")
            
            # Find duplicates
            unique_times, inverse_indices, counts = np.unique(time_pandas, return_inverse=True, return_counts=True)
            
            duplicated_times = unique_times[counts > 1]
            max_duplicates = counts.max()
            
            print(f"   Duplicated timestamps: {len(duplicated_times)}")
            print(f"   Max duplicates per timestamp: {max_duplicates}")
            
            if len(duplicated_times) > 0:
                print(f"\n📊 Duplication Pattern:")
                print(f"   Duplication counts: {np.bincount(counts)[1:]}")  # Skip count=0
                
                # Show first few duplicated timestamps
                print(f"\n   First 10 duplicated timestamps:")
                for i, dup_time in enumerate(duplicated_times[:10]):
                    dup_indices = np.where(time_pandas == dup_time)[0]
                    print(f"      {dup_time}: indices {dup_indices}")
                
                # Analyze variable availability at duplicated timestamps
                print(f"\n🔍 Variable Availability at Duplicated Timestamps:")
                
                # Pick a representative duplicated timestamp
                test_time = duplicated_times[0]
                test_indices = np.where(time_pandas == test_time)[0]
                
                print(f"\n   Testing timestamp: {test_time}")
                print(f"   Available at indices: {test_indices}")
                
                print(f"\n   Variable availability at each duplicate:")
                print(f"   {'Variable':<25} {'Index':<6} {'Has Data':<8} {'Sample Range'}")
                print("-" * 70)
                
                for var_name in ds.data_vars:
                    var_data = ds[var_name]
                    
                    if len(var_data.dims) >= 2:  # Has spatial dimensions
                        time_dim = var_data.dims[0]
                        
                        for idx in test_indices:
                            time_slice = var_data.isel({time_dim: idx})
                            values = time_slice.values
                            
                            has_data = np.any(np.isfinite(values))
                            
                            if has_data:
                                finite_values = values[np.isfinite(values)]
                                value_range = f"[{finite_values.min():.2f}, {finite_values.max():.2f}]"
                            else:
                                value_range = "All NaN"
                            
                            print(f"   {var_name:<25} {idx:<6} {has_data:<8} {value_range}")
                
                return {
                    'total_timesteps': len(time_values),
                    'unique_timestamps': len(unique_times),
                    'duplicated_timestamps': len(duplicated_times),
                    'max_duplicates': max_duplicates,
                    'duplication_factor': len(time_values) / len(unique_times),
                    'test_time': test_time,
                    'test_indices': test_indices
                }
            else:
                print(f"   ✅ No timestamp duplication found")
                return None
            
    except Exception as e:
        print(f"❌ Error analyzing timestamp duplication: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_variable_availability_matrix(file_path: str):
    """Create a matrix showing which variables have data at which timestamps"""
    
    print(f"\n📊 VARIABLE AVAILABILITY MATRIX")
    print("=" * 50)
    
    try:
        with xr.open_dataset(file_path) as ds:
            
            # Get time coordinate
            time_coord = None
            for coord in ds.coords:
                if 'time' in coord.lower():
                    time_coord = coord
                    break
            
            time_values = ds[time_coord].values
            time_pandas = pd.to_datetime(time_values)
            
            # Create availability matrix
            variables = list(ds.data_vars.keys())
            availability_matrix = np.zeros((len(variables), len(time_values)), dtype=bool)
            
            print(f"   Creating matrix: {len(variables)} variables × {len(time_values)} timesteps")
            
            for i, var_name in enumerate(variables):
                var_data = ds[var_name]
                
                if len(var_data.dims) >= 2:  # Has spatial dimensions
                    time_dim = var_data.dims[0]
                    
                    for t in range(len(time_values)):
                        time_slice = var_data.isel({time_dim: t})
                        availability_matrix[i, t] = np.any(np.isfinite(time_slice.values))
            
            # Analyze patterns
            print(f"\n   Variable availability summary:")
            print(f"   {'Variable':<25} {'Available Steps':<15} {'%':<8} {'Pattern'}")
            print("-" * 70)
            
            for i, var_name in enumerate(variables):
                available_steps = availability_matrix[i, :].sum()
                availability_percent = (available_steps / len(time_values)) * 100
                
                # Analyze pattern
                if availability_percent == 100:
                    pattern = "Full"
                elif availability_percent == 0:
                    pattern = "Empty"
                elif availability_percent > 66:
                    pattern = "Mostly full"
                elif availability_percent > 33:
                    pattern = "Partial"
                else:
                    pattern = "Mostly empty"
                
                print(f"   {var_name:<25} {available_steps:<15} {availability_percent:<8.1f} {pattern}")
            
            # Check for complementary patterns
            print(f"\n🔍 Complementary Pattern Analysis:")
            
            # Group variables by similar availability patterns
            pattern_groups = {}
            for i, var_name in enumerate(variables):
                pattern_key = tuple(availability_matrix[i, :])
                if pattern_key not in pattern_groups:
                    pattern_groups[pattern_key] = []
                pattern_groups[pattern_key].append(var_name)
            
            print(f"   Found {len(pattern_groups)} distinct availability patterns")
            
            for j, (pattern, vars_in_pattern) in enumerate(pattern_groups.items()):
                available_count = sum(pattern)
                total_count = len(pattern)
                coverage = (available_count / total_count) * 100
                
                print(f"\n   Pattern {j+1}: {coverage:.1f}% coverage ({available_count}/{total_count} steps)")
                print(f"      Variables: {vars_in_pattern}")
                
                if len(vars_in_pattern) > 1:
                    print(f"      ✅ Multiple variables share this pattern")
                else:
                    print(f"      ⚠️  Single variable with unique pattern")
            
            return {
                'availability_matrix': availability_matrix,
                'variables': variables,
                'time_values': time_values,
                'pattern_groups': pattern_groups
            }
            
    except Exception as e:
        print(f"❌ Error creating availability matrix: {e}")
        return None

def find_optimal_timestamp_selection(file_path: str, matrix_data: Dict):
    """Find the optimal way to select timestamps to get maximum data coverage"""
    
    print(f"\n🎯 OPTIMAL TIMESTAMP SELECTION")
    print("=" * 50)
    
    try:
        availability_matrix = matrix_data['availability_matrix']
        variables = matrix_data['variables']
        time_values = matrix_data['time_values']
        
        time_pandas = pd.to_datetime(time_values)
        unique_times = pd.unique(time_pandas)
        
        print(f"   Total timesteps: {len(time_values)}")
        print(f"   Unique timestamps: {len(unique_times)}")
        
        # For each unique timestamp, find the best representative index
        optimal_indices = []
        coverage_scores = []
        
        for unique_time in unique_times:
            # Find all indices with this timestamp
            matching_indices = np.where(time_pandas == unique_time)[0]
            
            if len(matching_indices) == 1:
                # No duplication, use the only index
                optimal_indices.append(matching_indices[0])
                coverage_scores.append(availability_matrix[:, matching_indices[0]].sum())
            else:
                # Multiple indices, find the one with maximum variable coverage
                best_idx = None
                best_coverage = 0
                
                for idx in matching_indices:
                    coverage = availability_matrix[:, idx].sum()
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_idx = idx
                
                optimal_indices.append(best_idx)
                coverage_scores.append(best_coverage)
        
        # Analyze the optimal selection
        optimal_indices = np.array(optimal_indices)
        coverage_scores = np.array(coverage_scores)
        
        print(f"\n   Optimal timestamp selection:")
        print(f"   Selected indices: {len(optimal_indices)}")
        print(f"   Average variables per timestamp: {coverage_scores.mean():.1f}")
        print(f"   Min variables per timestamp: {coverage_scores.min()}")
        print(f"   Max variables per timestamp: {coverage_scores.max()}")
        
        # Check overall coverage with optimal selection
        optimal_matrix = availability_matrix[:, optimal_indices]
        
        print(f"\n   Variable coverage with optimal selection:")
        print(f"   {'Variable':<25} {'Available Steps':<15} {'%':<8} {'Improvement'}")
        print("-" * 80)
        
        for i, var_name in enumerate(variables):
            original_coverage = availability_matrix[i, :].sum()
            optimal_coverage = optimal_matrix[i, :].sum()
            
            original_percent = (original_coverage / len(time_values)) * 100
            optimal_percent = (optimal_coverage / len(optimal_indices)) * 100
            
            improvement = optimal_percent - original_percent
            
            print(f"   {var_name:<25} {optimal_coverage:<15} {optimal_percent:<8.1f} {improvement:+.1f}%")
        
        return {
            'optimal_indices': optimal_indices,
            'coverage_scores': coverage_scores,
            'optimal_matrix': optimal_matrix,
            'unique_times': unique_times
        }
        
    except Exception as e:
        print(f"❌ Error finding optimal selection: {e}")
        return None

def generate_data_reconstruction_strategy(file_path: str):
    """Generate a strategy to reconstruct clean data from the problematic file"""
    
    print(f"\n🔧 DATA RECONSTRUCTION STRATEGY")
    print("=" * 50)
    
    # Run all analyses
    dup_analysis = analyze_timestamp_duplication(file_path)
    
    if dup_analysis is None:
        print(f"   ✅ No timestamp duplication - file may be clean")
        return
    
    matrix_data = create_variable_availability_matrix(file_path)
    
    if matrix_data is None:
        print(f"   ❌ Could not create availability matrix")
        return
    
    optimal_selection = find_optimal_timestamp_selection(file_path, matrix_data)
    
    if optimal_selection is None:
        print(f"   ❌ Could not find optimal selection")
        return
    
    # Generate reconstruction code
    print(f"\n💡 RECONSTRUCTION RECOMMENDATIONS:")
    
    duplication_factor = dup_analysis['duplication_factor']
    
    if duplication_factor > 2.5:
        print(f"   🚨 High duplication factor ({duplication_factor:.1f}x)")
        print(f"   💡 SOLUTION: Select optimal timestamps to remove duplicates")
        
        print(f"\n   📝 Python code to reconstruct clean dataset:")
        print(f"```python")
        print(f"import xarray as xr")
        print(f"import numpy as np")
        print(f"import pandas as pd")
        print(f"")
        print(f"# Load problematic dataset")
        print(f"ds = xr.open_dataset('{file_path}')")
        print(f"")
        print(f"# Get time coordinate and find optimal indices")
        print(f"time_values = ds.valid_time.values")
        print(f"time_pandas = pd.to_datetime(time_values)")
        print(f"unique_times = pd.unique(time_pandas)")
        print(f"")
        print(f"# Select best representative for each unique timestamp")
        print(f"optimal_indices = []")
        print(f"for unique_time in unique_times:")
        print(f"    matching_indices = np.where(time_pandas == unique_time)[0]")
        print(f"    ")
        print(f"    # Find index with most variables having data")
        print(f"    best_idx = None")
        print(f"    best_coverage = 0")
        print(f"    ")
        print(f"    for idx in matching_indices:")
        print(f"        coverage = 0")
        print(f"        for var_name in ds.data_vars:")
        print(f"            if len(ds[var_name].dims) >= 2:")
        print(f"                time_slice = ds[var_name].isel({{ds[var_name].dims[0]: idx}})")
        print(f"                if np.any(np.isfinite(time_slice.values)):")
        print(f"                    coverage += 1")
        print(f"        ")
        print(f"        if coverage > best_coverage:")
        print(f"            best_coverage = coverage")
        print(f"            best_idx = idx")
        print(f"    ")
        print(f"    optimal_indices.append(best_idx)")
        print(f"")
        print(f"# Create clean dataset")
        print(f"clean_ds = ds.isel(valid_time=optimal_indices)")
        print(f"")
        print(f"# Save reconstructed dataset")
        print(f"clean_ds.to_netcdf('era5_single_level_201910_CLEANED.nc')")
        print(f"print(f'Reconstructed dataset: {{len(clean_ds.valid_time)}} timesteps')")
        print(f"```")
        
    else:
        print(f"   ✅ Duplication factor is manageable ({duplication_factor:.1f}x)")
        print(f"   💡 SOLUTION: Simple deduplication may be sufficient")
    
    # Check if reconstruction would be worthwhile
    avg_coverage = optimal_selection['coverage_scores'].mean()
    total_variables = len(matrix_data['variables'])
    coverage_percent = (avg_coverage / total_variables) * 100
    
    print(f"\n   📊 Reconstruction Potential:")
    print(f"   Average variable coverage: {coverage_percent:.1f}%")
    
    if coverage_percent > 70:
        print(f"   ✅ Reconstruction would yield good coverage")
        print(f"   🚀 RECOMMEND: Proceed with reconstruction")
    elif coverage_percent > 50:
        print(f"   ⚠️  Reconstruction would yield moderate coverage")
        print(f"   💭 CONSIDER: Reconstruction vs re-download")
    else:
        print(f"   ❌ Reconstruction would yield poor coverage")
        print(f"   🔄 RECOMMEND: Re-download instead")

def main():
    """Main analysis function"""
    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_timestamp_duplication.py <single_level_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    print(f"🔍 ANALYZING TIMESTAMP DUPLICATION AND VARIABLE ALIGNMENT")
    print(f"=" * 80)
    
    generate_data_reconstruction_strategy(file_path)

if __name__ == "__main__":
    main()