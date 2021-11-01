import xarray as xr
from dask.diagnostics import ProgressBar
ProgressBar().register()

# merge the two gefs files - short (1-7d) and long (8-14d)
# note they have different spatial resolutions so average first
path = 'data/gefs_unprocessed/WCUS/'

# do not take spatial average
# instead resample to daily fhours (8 days in each ds)
# this is going to be slow - mfdataset dask uses lazy compute

ds = xr.open_mfdataset(path+'*.nc', combine='by_coords')
ds = ds.Total_precipitation.sel(fhour=ds.fhour[1:]) / 10 # skip 00 time, cm/day
ds = ds.resample(fhour='24H').sum(dim='fhour') # daily sum
ds = ds.sel(fhour=ds.fhour[:-1]) # remove last weird one
print(ds)
ds.to_netcdf(path + 'gefs_merged.nc')

