import numpy as np 
import cdsapi
import xarray as xr
import os

c = cdsapi.Client()
path = '/Volumes/jh-climate/era5/raw/'

# years = [str(y) for y in range(1979,2020)] # check later for 1950-
# months = ['%02d' % m for m in range(1,13)]
months = ['11', '12', '1', '2', '3'] # winter only
days = ['%02d' % d for d in range(1,32)]
times = ['%02d:00' % h for h in range(0,24)]

# testing setup pre-hard drive
# path = './'
# years = ['2019']
# months = ['12']
# days = ['07']
# times = ['01', '02']

# try this instead of by single years
chunks = [[str(y) for y in range(1979,1990)],
          [str(y) for y in range(1990,2000)],
          [str(y) for y in range(2000,2010)],
          [str(y) for y in range(2010,2020)]]

for i,years in enumerate(chunks):

    c.retrieve('reanalysis-era5-single-levels',
      { 'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'total_precipitation',
        'grid': '0.25/0.25',
        'year': years,
        'month': months,
        'day': days,
        'time': times,
        'area': '42/-123/38/-120', # North, West, South, East. Default: global
      }, path + 'download_target.nc')

    # upscale to daily
    ds = xr.open_dataset(path + 'download_target.nc')
    ds = ds.resample(time='D').sum(dim='time')
    ds.to_netcdf(path + 'era5_daily_target_precip_NCA_chunk%d.nc' % i)
    os.remove(path + 'download_target.nc')

