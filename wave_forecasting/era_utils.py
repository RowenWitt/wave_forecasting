import xarray as xr

def flatten_time_duplicates(ds):
    total_length = len(ds['valid_time'])
    segment_length = total_length / 3
    segments = [segment_length * i for i in range(3)]