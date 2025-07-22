"""
Quick diagnostic script to check your processed data
"""
import xarray as xr
import numpy as np
from pathlib import Path

def diagnose_data():
    """Diagnose the processed data file"""
    
    data_file = Path("data/processed_v1/enhanced_v1_era5_202101.nc")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"🔍 DIAGNOSING: {data_file.name}")
    print("=" * 60)
    
    try:
        ds = xr.open_dataset(data_file)
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   Variables: {len(ds.data_vars)}")
        print(f"   Coordinates: {list(ds.coords.keys())}")
        print(f"   Dimensions: {dict(ds.dims)}")
        
        print(f"\n🌊 WAVE VARIABLES CHECK:")
        wave_vars = ['swh', 'mwd', 'mwp']
        
        for var in wave_vars:
            if var in ds.data_vars:
                data = ds[var]
                print(f"   {var.upper()}:")
                print(f"      Shape: {data.shape}")
                print(f"      Range: {data.min().values:.3f} to {data.max().values:.3f}")
                print(f"      Mean: {data.mean().values:.3f}")
                print(f"      Non-zero count: {np.count_nonzero(data.values):,}")
                print(f"      NaN count: {np.isnan(data.values).sum():,}")
                
                # Check first timestep
                if len(data.shape) >= 3:
                    first_timestep = data.isel({list(data.dims)[0]: 0})
                    print(f"      First timestep range: {first_timestep.min().values:.3f} to {first_timestep.max().values:.3f}")
            else:
                print(f"   ❌ {var.upper()}: NOT FOUND")
        
        print(f"\n🌡️  ATMOSPHERIC VARIABLES CHECK:")
        atmo_vars = ['u10', 'v10', 'msl', 'sst', 'tp']
        
        for var in atmo_vars:
            if var in ds.data_vars:
                data = ds[var]
                print(f"   {var.upper()}: range {data.min().values:.3f} to {data.max().values:.3f}")
            else:
                print(f"   ❌ {var.upper()}: NOT FOUND")
        
        print(f"\n❄️  CLIMATE ANOMALIES CHECK:")
        anomaly_vars = [var for var in ds.data_vars if 'anomaly' in var]
        
        for var in anomaly_vars:
            data = ds[var]
            print(f"   {var}: range {data.min().values:.3f} to {data.max().values:.3f}")
        
        print(f"\n🌊 BATHYMETRY CHECK:")
        if 'ocean_depth' in ds.data_vars:
            bathy = ds['ocean_depth']
            print(f"   Ocean depth: range {bathy.min().values:.1f} to {bathy.max().values:.1f}m")
            print(f"   Ocean points: {(bathy > 0).sum().values:,}")
            print(f"   Land points: {(bathy == 0).sum().values:,}")
        else:
            print(f"   ❌ Ocean depth: NOT FOUND")
        
        print(f"\n📍 COORDINATE CHECK:")
        spatial_coords = [coord for coord in ds.coords if coord in ['lat', 'lon', 'latitude', 'longitude']]
        for coord in spatial_coords:
            coord_data = ds[coord]
            print(f"   {coord}: {len(coord_data)} points, range {coord_data.min().values:.2f} to {coord_data.max().values:.2f}")
        
        print(f"\n⏰ TIME CHECK:")
        time_coords = [coord for coord in ds.coords if 'time' in coord.lower()]
        for coord in time_coords:
            coord_data = ds[coord]
            print(f"   {coord}: {len(coord_data)} timesteps")
            if len(coord_data) > 0:
                print(f"      First: {coord_data[0].values}")
                print(f"      Last: {coord_data[-1].values}")
        
        ds.close()
        
        print(f"\n✅ DIAGNOSIS COMPLETE")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    diagnose_data()