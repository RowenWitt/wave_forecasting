import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load the data
atmo_data = xr.open_dataset('era_data/era5_atmo_202301.nc')
wave_data = xr.open_dataset('era_data/era5_waves_202301.nc')

print("Atmospheric data shape:", atmo_data.dims)
print("Wave data shape:", wave_data.dims)

# Quick visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Wind speed
wind_speed = np.sqrt(atmo_data.u10**2 + atmo_data.v10**2)
wind_speed.isel(valid_time=0).plot(ax=axes[0], cmap='viridis')
axes[0].set_title('Wind Speed (m/s)')

# Wave height
wave_data.swh.isel(valid_time=0).plot(ax=axes[1], cmap='plasma')
axes[1].set_title('Significant Wave Height (m)')

plt.tight_layout()
plt.show()