import numpy as np 
import xarray as xr
import pickle

# inputs: GEFS west coast, individual days, 1-10d leads
# outputs: ERA5 target precip, spatial avg over the sacramento

def anynan(a):
  return np.isnan(a).any()

sd,ed = (2011,2019) # training 1985-2010 
def test_mask(ds):
  return (ds['time.year'] >= sd) & (ds['time.year'] <= ed)

def ttsplit(X,y,ix):
  X_train = X[~ix,:,:,:]
  X_test = X[ix,:,:,:]
  y_train = y[~ix]
  y_test = y[ix]
  return X_train, y_train, X_test, y_test

def remove_nans(X,y):
  yix = ~np.isnan(y)
  return X[yix,:,:,:], y[yix]

def print_checks(X,y):
  print('X shape: ', X.shape)
  print('y shape: ', y.shape)
  print('NaN Check-- ', 'X: ', anynan(X), ', y: ', anynan(y))

# feature data
path = '/data/'
ds = xr.open_dataset(path + 'gefs/WCUS/gefs_merged.nc')
# shift feature data
ds = xr.concat([ds.sel(fhour=hour).shift(time=i) for i,hour in enumerate(ds.fhour)],'fhour')
ds = ds.sel(time=slice('1985-01-01', '2019-12-31'))

# switch from xarray to numpy array
X = ds.Total_precipitation.values
X = np.rollaxis(X, 0, 4) # switch channels 0 to channels last 

# target vector - don't impose quantiles yet
ds_target = xr.open_dataset(path + 'era5_daily_target_precip_NCA.nc')
ds_target = ds_target.sel(time=slice('1985-01-01', '2019-12-31'))
y = ds_target.tp.values.astype(float) # precip values shape=(time,)


# train test split
print('ttsplit...')
X_train, y_train, X_test, y_test = ttsplit(X, y, test_mask(ds))

print('Checks before removing nans')
print_checks(X_train, y_train)
print_checks(X_test, y_test)

# remove nans due to seasonality (keep NDJFM within forecast lead)
X_train, y_train = remove_nans(X_train, y_train)
X_test, y_test = remove_nans(X_test, y_test)

# shape check: X (time, lat, lon, channels), y (time,)
print('Checks after removing nans')
print_checks(X_train, y_train)
print_checks(X_test, y_test)

# 4/1/20: save y as continuous, add thresholds later
# don't know why but np.save doesn't work anymore
data = (X_train, y_train, X_test, y_test)
pickle.dump(data, open(path+'tensor/gefs_mos_daily_tensor.pkl', 'wb'))

