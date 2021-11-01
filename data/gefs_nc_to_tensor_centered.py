import numpy as np 
import xarray as xr
import pickle
# import pandas as pd

# inputs: GEFS west coast, individual days, 1-14d leads
# outputs: ERA5 target precip, spatial avg over the sacramento
 
def anynan(a):
  return np.isnan(a).any()

test_sd,test_ed = (2011,2019) # training 1985-2010 

def test_mask(ds):
  return (ds['time.year'] >= test_sd) & (ds['time.year'] <= test_ed)

def ttsplit(X,y,ix):
  X_train = X[~ix,:,:,:]
  X_test = X[ix,:,:,:]
  y_train = y[~ix]
  y_test = y[ix]
  return X_train, y_train, X_test, y_test

def remove_nans(X,y):
  yix = ~np.isnan(y)
  return X[yix,:,:,:], y[yix]

def is_ndjfm(month):
    return (month <= 3) | (month >= 11)

def print_checks(X,y):
  print('X shape: ', X.shape)
  print('y shape: ', y.shape)
  print('NaN Check-- ', 'X: ', anynan(X), ', y: ', anynan(y))

# feature data
path = '/data/'
precip = xr.open_dataset(path + 'gefs/WCUS/gefs_merged.nc')
precip = xr.concat([precip.sel(fhour=hour).shift(time=i) for i,hour in enumerate(precip.fhour)],'fhour')
precip = precip.sel(time=slice('1985-01-01', '2019-12-31'))
precip = precip.sel(time=is_ndjfm(precip['time.month']))


# switch from xarray to numpy array
X = precip.to_array().values # shape of X (1, 14, 5293, 49, 45)
X = np.rollaxis(X, 2) # move 3rd element to first
X = np.rollaxis(X, 1, 5)
X = np.rollaxis(X, 1, 5) # adjust shape to (5293, 49, 45, 1, 14)


# target vector - don't impose quantiles yet
ds_target = xr.open_dataset(path + 'era5_daily_target_precip_NCA.nc')
ds_target = ds_target.sel(time=slice('1985-01-01', '2019-12-31'))
y = ds_target.tp.values.astype(float) # precip values shape=(time,)=（12783,）
y = y[~np.isnan(y)] # removed nan from y so that shape=(5293,)

    
# train test split
print('ttsplit...')
X_train, y_train, X_test, y_test = ttsplit(X, y, test_mask(precip))

# standardization
# max_lead = 14
# for l in range(max_lead):
#     precip_mean = X_train[:,:,:,0,l].mean()
#     precip_std = np.std(X_train[:,:,:,0,l])
#     X_train[:,:,:,0,l]=(X_train[:,:,:,0,l]-precip_mean)/precip_std
#     X_test[:,:,:,0,l]=(X_test[:,:,:,0,l]-precip_mean)/precip_std

# center
max_lead = 14
for l in range(max_lead):
    precip_mean = X_train[:,:,:,0,l].mean()
    X_train[:,:,:,0,l]=X_train[:,:,:,0,l]-precip_mean
    X_test[:,:,:,0,l]=X_test[:,:,:,0,l]-precip_mean

# shape check: X (time, lat, lon, channels), y (time,)
print('Checking nan and dimensions before saving...')
print_checks(X_train, y_train)
print_checks(X_test, y_test)

# 4/1/20: save y as continuous, add thresholds later
# don't know why but np.save doesn't work anymore
# combined_data = (X, y)
data = (X_train, y_train, X_test, y_test)
pickle.dump(data, open(path+'tensor/gefs_mos_daily_tensor_precip_center.pkl', 'wb'))
