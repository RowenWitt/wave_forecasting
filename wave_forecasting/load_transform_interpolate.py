import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from build_mesh import IcosahedralMesh

def load_and_inspect_datasets():
    """Load all datasets and understand their structure"""
    
    print("Loading ERA5 atmospheric data (0.25°)...")
    era5_atmo = xr.open_dataset('era_data/era5_atmo_202301.nc')
    
    print("Loading ERA5 wave data (0.5°)...")
    era5_waves = xr.open_dataset('era_data/era5_waves_202301.nc')
    
    print("Loading GEBCO bathymetry (15 arcsec)...")
    gebco = xr.open_dataset('gebco_data/gebco_2023_sub_ice_n60.0_s10.0_w120.0_e240.0.nc')
    
    # Print data structures
    print("\n" + "="*50)
    print("ERA5 ATMOSPHERIC (0.25°)")
    print("="*50)
    print(f"Dimensions: {era5_atmo.dims}")
    print(f"Variables: {list(era5_atmo.data_vars)}")
    print(f"Coordinate ranges:")
    print(f"  Lat: {era5_atmo.latitude.min().values:.2f} to {era5_atmo.latitude.max().values:.2f}")
    print(f"  Lon: {era5_atmo.longitude.min().values:.2f} to {era5_atmo.longitude.max().values:.2f}")
    print(f"  Time: {era5_atmo.valid_time.min().values} to {era5_atmo.valid_time.max().values}")
    
    print("\n" + "="*50)
    print("ERA5 WAVES (0.5°)")
    print("="*50)
    print(f"Dimensions: {era5_waves.dims}")
    print(f"Variables: {list(era5_waves.data_vars)}")
    print(f"Coordinate ranges:")
    print(f"  Lat: {era5_waves.latitude.min().values:.2f} to {era5_waves.latitude.max().values:.2f}")
    print(f"  Lon: {era5_waves.longitude.min().values:.2f} to {era5_waves.longitude.max().values:.2f}")
    
    print("\n" + "="*50)
    print("GEBCO BATHYMETRY (15 arcsec)")
    print("="*50)
    print(f"Dimensions: {gebco.dims}")
    print(f"Variables: {list(gebco.data_vars)}")
    
    # Figure out GEBCO coordinate and variable names
    lat_coord = 'lat' if 'lat' in gebco.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in gebco.coords else 'longitude'
    depth_var = 'elevation' if 'elevation' in gebco else 'z'
    
    print(f"Coordinate names: {lat_coord}, {lon_coord}")
    print(f"Elevation variable: {depth_var}")
    print(f"  Lat: {gebco[lat_coord].min().values:.2f} to {gebco[lat_coord].max().values:.2f}")
    print(f"  Lon: {gebco[lon_coord].min().values:.2f} to {gebco[lon_coord].max().values:.2f}")
    print(f"  Elevation range: {gebco[depth_var].min().values:.0f} to {gebco[depth_var].max().values:.0f}m")
    
    return era5_atmo, era5_waves, gebco

# Load everything
# era5_atmo, era5_waves, gebco = load_and_inspect_datasets()

def process_gebco_for_waves(gebco):
    """Convert GEBCO to wave-modeling features"""
    
    # Standardize coordinate names
    lat_coord = 'lat' if 'lat' in gebco.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in gebco.coords else 'longitude' 
    depth_var = 'elevation' if 'elevation' in gebco else 'z'
    
    # Rename for consistency
    gebco_std = gebco.rename({
        depth_var: 'z',
        lat_coord: 'latitude',
        lon_coord: 'longitude'
    })
    
    print("Processing bathymetry features...")
    
    # Ocean depth (positive values, NaN on land)
    ocean_depth = -gebco_std.z.where(gebco_std.z < 0)
    
    # Land-sea mask (1 = ocean, 0 = land)
    land_sea_mask = (gebco_std.z < 0).astype(float)
    
    # Compute gradients (critical for wave refraction)
    print("Computing depth gradients...")
    depth_grad_lat = ocean_depth.differentiate('latitude')
    depth_grad_lon = ocean_depth.differentiate('longitude')
    depth_gradient = np.sqrt(depth_grad_lat**2 + depth_grad_lon**2)
    
    # Wave-relevant depth categories
    shallow_water = (ocean_depth < 50).astype(float)      # Wave speed = sqrt(g*h)
    shelf_water = ((ocean_depth >= 50) & (ocean_depth < 500)).astype(float)
    deep_water = (ocean_depth >= 500).astype(float)       # "Infinite" depth for waves
    
    # Steep bottom features (seamounts, ridges, shelf breaks)
    steep_bottom = (depth_gradient > depth_gradient.quantile(0.95)).astype(float)
    
    # Create processed dataset
    processed_gebco = xr.Dataset({
        'z': gebco_std.z,
        'ocean_depth': ocean_depth,
        'land_sea_mask': land_sea_mask,
        'depth_gradient': depth_gradient,
        'shallow_water_mask': shallow_water,
        'shelf_water_mask': shelf_water,
        'deep_water_mask': deep_water,
        'steep_bottom_mask': steep_bottom,
    })
    
    # Save processed version
    processed_gebco.to_netcdf('gebco_data/processed_gebco_wave_features.nc')
    print("✅ GEBCO processed and saved!")
    
    return processed_gebco

# Process GEBCO
# processed_gebco = process_gebco_for_waves(gebco)

from scipy.interpolate import RegularGridInterpolator
import numpy as np

class MultiResolutionInterpolator:
    """Handles interpolation from multiple source grids to mesh nodes"""
    
    def __init__(self, era5_atmo, era5_waves, gebco_processed):
        self.era5_atmo = era5_atmo
        self.era5_waves = era5_waves 
        self.gebco = gebco_processed
        
        # Create interpolators for each dataset
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Pre-build interpolation functions for efficiency"""
        
        print("Setting up multi-resolution interpolators...")
        
        # ERA5 atmospheric interpolators (0.25°)
        self.atmo_interpolators = {}
        for var in ['u10', 'v10']:  # Add more variables as needed
            self.atmo_interpolators[var] = RegularGridInterpolator(
                (self.era5_atmo.latitude.values, self.era5_atmo.longitude.values),
                self.era5_atmo[var].isel(valid_time=0).values,  # We'll handle time separately
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        
        # ERA5 wave interpolators (0.5°)  
        self.wave_interpolators = {}
        for var in ['swh', 'mwd', 'mwp']:  # Add more wave variables
            if var in self.era5_waves:
                self.wave_interpolators[var] = RegularGridInterpolator(
                    (self.era5_waves.latitude.values, self.era5_waves.longitude.values),
                    self.era5_waves[var].isel(valid_time=0).values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
        
        # GEBCO bathymetry interpolators (15 arcsec) - static, so no time dimension
        self.bathy_interpolators = {}
        for var in ['ocean_depth', 'land_sea_mask', 'depth_gradient', 
                   'shallow_water_mask', 'deep_water_mask']:
            self.bathy_interpolators[var] = RegularGridInterpolator(
                (self.gebco.latitude.values, self.gebco.longitude.values),
                self.gebco[var].values,
                method='linear',
                bounds_error=False,
                fill_value=0.0  # Default to land/no-data for bathymetry
            )
        
        print("✅ Interpolators ready!")
    
    def interpolate_to_points(self, lats, lons, time_idx=0):
        """Interpolate all variables to given lat/lon points"""
        
        # Update time-dependent interpolators
        self._update_time_interpolators(time_idx)
        
        # Create coordinate pairs for interpolation
        points = np.column_stack([lats.ravel(), lons.ravel()])
        
        # Interpolate from each dataset
        results = {}
        
        # Atmospheric variables (0.25°)
        for var, interpolator in self.atmo_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        # Wave variables (0.5°)
        for var, interpolator in self.wave_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        # Bathymetry variables (15 arcsec) - static
        for var, interpolator in self.bathy_interpolators.items():
            results[var] = interpolator(points).reshape(lats.shape)
        
        return results
    
    def _update_time_interpolators(self, time_idx):
        """Update interpolators with new time slice"""
        
        # Update atmospheric interpolators
        for var in self.atmo_interpolators:
            self.atmo_interpolators[var].values = self.era5_atmo[var].isel(valid_time=time_idx).values
        
        # Update wave interpolators
        for var in self.wave_interpolators:
            self.wave_interpolators[var].values = self.era5_waves[var].isel(valid_time=time_idx).values

# Create the multi-resolution interpolator
# interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, processed_gebco)

def test_multi_resolution_interpolation():
    """Test that our multi-resolution interpolation works"""
    
    # Test points across the domain
    test_lats = np.array([20.0, 35.0, 50.0])  # 3 test latitudes
    test_lons = np.array([140.0, 180.0, 220.0])  # 3 test longitudes
    
    # Create mesh of test points
    test_lon_grid, test_lat_grid = np.meshgrid(test_lons, test_lats)
    
    print("Testing multi-resolution interpolation...")
    
    # Interpolate all variables to test points
    results = interpolator.interpolate_to_points(test_lat_grid, test_lon_grid, time_idx=0)
    
    print("\nInterpolation results:")
    print("="*30)
    
    for var, values in results.items():
        print(f"{var:20s}: {values.ravel()}")
    
    # Quick visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    plot_vars = ['u10', 'swh', 'ocean_depth', 'land_sea_mask', 'depth_gradient', 'shallow_water_mask']
    
    for i, var in enumerate(plot_vars[:6]):
        if var in results:
            im = axes[i].imshow(results[var], extent=[test_lons.min(), test_lons.max(), 
                                                    test_lats.min(), test_lats.max()])
            axes[i].set_title(f'{var}')
            plt.colorbar(im, ax=axes[i])
    
    # plt.tight_layout()
    # plt.show()
    
    print("✅ Multi-resolution interpolation working!")

# Test it
# test_multi_resolution_interpolation()

def benchmark_interpolation_performance():
    """Check how fast our multi-resolution interpolation is"""
    
    import time
    
    # Simulate mesh-sized interpolation (40k nodes)
    n_points = 40000
    test_lats = np.random.uniform(10, 60, n_points)
    test_lons = np.random.uniform(120, 240, n_points)
    
    print(f"Benchmarking interpolation for {n_points} points...")
    
    start_time = time.time()
    results = interpolator.interpolate_to_points(test_lats, test_lons, time_idx=0)
    end_time = time.time()
    
    print(f"⏱️  Interpolation time: {end_time - start_time:.2f} seconds")
    print(f"📊 Rate: {n_points / (end_time - start_time):.0f} points/second")
    
    # Memory usage
    total_memory = sum(arr.nbytes for arr in results.values()) / 1e6
    print(f"💾 Memory usage: {total_memory:.1f} MB")

# Benchmark it
# benchmark_interpolation_performance()


class MeshDataLoader:
    """Connects icosahedral mesh to multi-resolution ERA5/GEBCO data"""
    
    def __init__(self, mesh, interpolator):
        self.mesh = mesh
        self.interpolator = interpolator
        
        # Get mesh coordinates
        self.mesh_lats, self.mesh_lons = mesh.vertices_to_lat_lon()
        
        # Focus on North Pacific region
        self.pacific_mask = ((self.mesh_lats >= 10) & (self.mesh_lats <= 60) & 
                           (self.mesh_lons >= 120) & (self.mesh_lons <= 240))
        
        self.pacific_node_indices = np.where(self.pacific_mask)[0]
        
        print(f"Mesh data loader ready:")
        print(f"  Pacific nodes: {len(self.pacific_node_indices)}")
    
    def load_mesh_features(self, time_idx=0):
        """Load all features onto mesh nodes"""
        
        # Get coordinates for Pacific nodes only
        pacific_lats = self.mesh_lats[self.pacific_mask]
        pacific_lons = self.mesh_lons[self.pacific_mask]
        
        # Interpolate all variables to mesh nodes
        interpolated_data = self.interpolator.interpolate_to_points(
            pacific_lats, pacific_lons, time_idx=time_idx
        )
        
        # Convert to feature matrix [nodes, features]
        feature_names = list(interpolated_data.keys())
        n_nodes = len(pacific_lats)
        n_features = len(feature_names)
        
        feature_matrix = np.zeros((n_nodes, n_features))
        
        for i, feature_name in enumerate(feature_names):
            feature_matrix[:, i] = interpolated_data[feature_name].ravel()
        
        return {
            'features': feature_matrix,
            'feature_names': feature_names,
            'node_indices': self.pacific_node_indices,
            'coordinates': np.column_stack([pacific_lats, pacific_lons])
        }



# # Connect mesh to data
# mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
# mesh_loader = MeshDataLoader(mesh, interpolator)

# # Test loading features
# test_features = mesh_loader.load_mesh_features(time_idx=0)
# print(f"✅ Loaded features: {test_features['features'].shape}")
# print(f"Feature names: {test_features['feature_names']}")