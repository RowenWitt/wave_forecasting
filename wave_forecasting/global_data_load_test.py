# test_data_loading.py
"""
Test data loading for the simplified global model
Validates that smart-merged files work correctly
"""

import xarray as xr
import numpy as np
from pathlib import Path
import torch
from simplified_global_model import SimplifiedDataset, SimplifiedGlobalConfig, EXPECTED_VARIABLES

def inspect_smart_merged_file(file_path: str):
    """Inspect the structure of a smart-merged file"""
    
    print(f"🔍 INSPECTING: {Path(file_path).name}")
    print("=" * 60)
    
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"📊 Dataset Overview:")
            print(f"   Dimensions: {dict(ds.dims)}")
            print(f"   Coordinates: {list(ds.coords.keys())}")
            print(f"   Variables: {len(ds.data_vars)}")
            print(f"   File size: {Path(file_path).stat().st_size / (1024**2):.1f} MB")
            
            # Check time range
            if 'valid_time' in ds.coords:
                time_vals = ds.valid_time.values
                print(f"   Time range: {time_vals[0]} to {time_vals[-1]}")
                print(f"   Time steps: {len(time_vals)}")
            
            # Show all variables
            print(f"\n📋 All Variables:")
            for var in sorted(ds.data_vars.keys()):
                var_info = ds[var]
                print(f"   {var:25} {str(var_info.dims):25} {var_info.shape}")
            
            # Check for expected variables
            print(f"\n✅ Expected Variable Check:")
            found_count = 0
            for expected_var, possible_names in EXPECTED_VARIABLES.items():
                found = False
                found_name = None
                for possible_name in possible_names:
                    if possible_name in ds.data_vars:
                        found = True
                        found_name = possible_name
                        found_count += 1
                        break
                
                status = "✅" if found else "❌"
                name_info = f"({found_name})" if found else ""
                print(f"   {status} {expected_var:15} {name_info}")
            
            print(f"\n📈 Summary: Found {found_count}/{len(EXPECTED_VARIABLES)} expected variables")
            
            # Check for nested pressure variables (should be flat)
            print(f"\n🔍 Checking for nested pressure dimensions:")
            nested_vars = []
            for var in ds.data_vars:
                if any(dim in ds[var].dims for dim in ['pressure_level', 'level', 'plev']):
                    nested_vars.append(var)
            
            if nested_vars:
                print(f"   ⚠️  Found nested variables: {nested_vars}")
                print(f"   👉 These should be flattened by smart merger")
            else:
                print(f"   ✅ No nested pressure variables (good!)")
            
            # Sample data quality check
            print(f"\n🔬 Data Quality Check:")
            for var_name in ['u10', 'swh', 'u850']:  # Sample key variables
                possible_names = EXPECTED_VARIABLES.get(var_name, [var_name])
                for possible_name in possible_names:
                    if possible_name in ds.data_vars:
                        data = ds[possible_name].values
                        nan_count = np.isnan(data).sum()
                        inf_count = np.isinf(data).sum()
                        min_val = np.nanmin(data)
                        max_val = np.nanmax(data)
                        
                        print(f"   {possible_name:15} NaN: {nan_count:8,} Inf: {inf_count:8,} Range: [{min_val:.2f}, {max_val:.2f}]")
                        break
            
    except Exception as e:
        print(f"❌ Error inspecting file: {e}")

def test_dataset_creation(file_path: str):
    """Test creating a dataset from the file"""
    
    print(f"\n🧪 TESTING DATASET CREATION")
    print("=" * 60)
    
    config = SimplifiedGlobalConfig(
        sequence_length=6,
        spatial_subsample=8,  # Use subsampling for testing
        batch_size=2
    )
    
    try:
        # Create dataset
        print(f"📦 Creating dataset...")
        dataset = SimplifiedDataset([file_path], config, is_validation=False)
        
        print(f"   ✅ Dataset created successfully")
        print(f"   📊 Total sequences: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"   ❌ No sequences available!")
            return
        
        # Test getting a sample
        print(f"\n🔬 Testing sample extraction...")
        sample = dataset[0]
        
        input_tensor = sample['input']
        target_tensor = sample['target']
        
        print(f"   ✅ Sample extracted successfully")
        print(f"   📊 Input shape: {input_tensor.shape}")
        print(f"   📊 Target shape: {target_tensor.shape}")
        print(f"   📊 Input dtype: {input_tensor.dtype}")
        print(f"   📊 Target dtype: {target_tensor.dtype}")
        
        # Check for NaN/inf values
        input_nan = torch.isnan(input_tensor).sum().item()
        input_inf = torch.isinf(input_tensor).sum().item()
        target_nan = torch.isnan(target_tensor).sum().item()
        target_inf = torch.isinf(target_tensor).sum().item()
        
        print(f"   🔍 Input NaN/Inf: {input_nan}/{input_inf}")
        print(f"   🔍 Target NaN/Inf: {target_nan}/{target_inf}")
        
        # Show data ranges
        print(f"   📏 Input range: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}]")
        print(f"   📏 Target range: [{target_tensor.min().item():.3f}, {target_tensor.max().item():.3f}]")
        
        # Test multiple samples
        print(f"\n🔄 Testing multiple samples...")
        sample_shapes = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            sample_shapes.append((sample['input'].shape, sample['target'].shape))
        
        # Check shape consistency
        first_input_shape = sample_shapes[0][0]
        first_target_shape = sample_shapes[0][1]
        
        shapes_consistent = all(
            s[0] == first_input_shape and s[1] == first_target_shape 
            for s in sample_shapes
        )
        
        if shapes_consistent:
            print(f"   ✅ All samples have consistent shapes")
        else:
            print(f"   ⚠️  Sample shapes are inconsistent:")
            for i, (input_shape, target_shape) in enumerate(sample_shapes):
                print(f"      Sample {i}: {input_shape} -> {target_shape}")
        
        # Test DataLoader creation
        print(f"\n🔄 Testing DataLoader...")
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        batch = next(iter(dataloader))
        print(f"   ✅ DataLoader batch created")
        print(f"   📊 Batch input shape: {batch['input'].shape}")
        print(f"   📊 Batch target shape: {batch['target'].shape}")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

def test_with_model(file_path: str):
    """Test loading data and running through model"""
    
    print(f"\n🤖 TESTING WITH MODEL")
    print("=" * 60)
    
    config = SimplifiedGlobalConfig(
        sequence_length=6,
        spatial_subsample=16,  # Higher subsampling for model test
        batch_size=1  # Single batch for testing
    )
    
    try:
        # Create dataset and model
        print(f"📦 Creating dataset and model...")
        dataset = SimplifiedDataset([file_path], config, is_validation=False)
        
        if len(dataset) == 0:
            print(f"❌ No data available for model test")
            return
        
        from simplified_global_model import SimplifiedSpatioTemporalGNN
        model = SimplifiedSpatioTemporalGNN(config)
        
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        sample = dataset[0]
        input_tensor = sample['input'].unsqueeze(0)  # Add batch dimension
        
        print(f"   📊 Testing forward pass...")
        print(f"   📊 Input shape: {input_tensor.shape}")
        
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   📊 Output shape: {predictions.shape}")
        print(f"   📏 Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
        
        # Check output structure
        if predictions.shape[-1] == 4:
            print(f"   ✅ Correct output format: [SWH, MWD_cos, MWD_sin, MWP]")
        else:
            print(f"   ⚠️  Unexpected output dimensions: {predictions.shape[-1]} (expected 4)")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main testing function"""
    
    print("🧪 DATA LOADING TEST SUITE")
    print("=" * 80)
    
    # Find test file
    data_dir = Path("data/era5_global")
    
    test_files = []
    for pattern in ["era5_smart_joined_*.nc", "era5_joined_*.nc"]:
        test_files.extend(data_dir.glob(pattern))
    
    if not test_files:
        print("❌ No test files found!")
        print("   Expected files in data/era5_global/:")
        print("   - era5_smart_joined_*.nc (preferred)")
        print("   - era5_joined_*.nc (fallback)")
        print("\n   Run the merger first:")
        print("   python smart_pressure_merger.py data/era5_global")
        return
    
    # Use first available file
    test_file = str(test_files[0])
    print(f"🎯 Testing with: {Path(test_file).name}")
    
    # Run tests
    inspect_smart_merged_file(test_file)
    test_dataset_creation(test_file)
    test_with_model(test_file)
    
    print(f"\n🎉 DATA LOADING TEST COMPLETE!")
    print(f"   If all tests passed, you can run training with:")
    print(f"   python simplified_global_model.py")

if __name__ == "__main__":
    main()