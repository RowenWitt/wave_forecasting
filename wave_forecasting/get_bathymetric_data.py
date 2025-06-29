import requests
import xarray as xr

def download_etopo_north_pacific():
    """Download ETOPO data for North Pacific region"""
    
    # ETOPO 2022 60-second bedrock URL
    base_url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/ETOPO_2022_v1_60s_N90W180_bed.nc"
    
    # Open remote dataset
    etopo_global = xr.open_dataset(base_url)
    
    # Subset to North Pacific (same bounds as ERA5)
    etopo_pacific = etopo_global.sel(
        lat=slice(10, 60),    # 10°N to 60°N
        lon=slice(120, 240)   # 120°E to 240°E (crosses dateline)
    )
    
    # Download subset
    etopo_pacific.load()  # Force download
    etopo_pacific.to_netcdf('etopo_north_pacific.nc')
    
    print(f"Downloaded bathymetry: {etopo_pacific.z.shape}")
    return etopo_pacific

# Download it
# bathymetry = download_etopo_north_pacific()

def download_gebco_pacific():
    """GEBCO is more reliable than ETOPO for programmatic access"""
    
    # GEBCO OPeNDAP server (usually works better)
    gebco_url = "https://www.bodc.ac.uk/thredds/dodsC/GEBCO_2024/GEBCO_2024.nc"
    
    try:
        print("Connecting to GEBCO server...")
        gebco_global = xr.open_dataset(gebco_url)
        print("X")
        # GEBCO uses different coordinate names
        # Rename to match our expectations
        if 'lat' in gebco_global.coords:
            lat_name, lon_name = 'lat', 'lon'
        elif 'latitude' in gebco_global.coords:
            lat_name, lon_name = 'latitude', 'longitude'
        else:
            lat_name, lon_name = list(gebco_global.coords)[:2]
        
        print(f"GEBCO dimensions: {gebco_global.dims}")
        print(f"GEBCO variables: {list(gebco_global.data_vars)}")
        
        # Subset to North Pacific
        gebco_pacific = gebco_global.sel({
            lat_name: slice(10, 60),
            lon_name: slice(120, 240)
        })
        
        print("Downloading GEBCO subset...")
        gebco_pacific.load()
        gebco_pacific.to_netcdf('gebco_north_pacific.nc')
        
        print("✅ GEBCO download successful!")
        return gebco_pacific
        
    except Exception as e:
        print(f"❌ GEBCO failed: {e}")
        return None

# Try GEBCO
# gebco_data = download_gebco_pacific()