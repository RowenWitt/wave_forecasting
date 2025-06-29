import cdsapi

q1 = ['01', '02', '03']
q2 = ['04', '05', '06']
q3 = ['07', '08', '09']
q4 = ['10', '11', '12']

def download_era5_atmospheric(output_file='era5_atmo_2023.nc'):
    c = cdsapi.Client()
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind', 
                'mean_sea_level_pressure',
            ],
            'year': '2023',
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],
            'area': [60, 120, 10, 240],  # North, West, South, East
            'format': 'netcdf',
        },
        output_file
    )


def download_era5_waves(output_file='era5_waves_2023.nc'):
    c = cdsapi.Client()
    
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'significant_height_of_combined_wind_waves_and_swell',
                'mean_wave_direction',
                'mean_wave_period',
                'significant_height_of_wind_waves',
                'mean_direction_of_wind_waves', 
                'mean_period_of_wind_waves',
            ],
            'year': '2023',
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],
            'area': [60, 120, 10, 240],
            'format': 'netcdf',
        },
        output_file
    )


def chunk_download_by_month(year=2023, months=['01', '02', '03']):
    """Download month by month to avoid limits"""
    c = cdsapi.Client()

    for month in months:
        print(f"Downloading {month}/{year}...")
        
        # Atmospheric
        c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
            'year': str(year),
            'month': month,
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],
            'area': [60, 120, 10, 240],
            'format': 'netcdf',
        }, f'era5_atmo_{year}{month}.nc')
        
        # Wave data
        c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis', 
            'variable': ['significant_height_of_combined_wind_waves_and_swell',
                        'mean_wave_direction', 'mean_wave_period'],
            'year': str(year),
            'month': month,
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24, 6)],
            'area': [60, 120, 10, 240],
            'format': 'netcdf',
        }, f'era5_waves_{year}{month}.nc')
        
        print(f"Month {month} complete")
# Run download
# download_era5_atmospheric()

import time

def download_with_retry(retrieve_params, output_file, max_retries=3):
    """Handle queue limits gracefully"""
    
    for attempt in range(max_retries):
        try:
            c.retrieve('reanalysis-era5-single-levels', retrieve_params, output_file)
            print(f"✅ Successfully downloaded {output_file}")
            return
        except Exception as e:
            if "queue" in str(e).lower():
                wait_time = 300 * (attempt + 1)  # Exponential backoff
                print(f"⏳ Queue full, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    
    raise Exception(f"Failed to download after {max_retries} attempts")