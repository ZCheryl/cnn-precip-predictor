import numpy as np 
import pandas as pd
import pickle
import xarray as xr

def is_NDJFM(month):
  return (month >= 11) | (month <= 3)

def time_mask(ds):
  return (is_NDJFM(ds['time.month']) & 
         (ds['time.year'] >= 1985) & # whole period. test is 2011.
         (ds['time.year'] <= 2019)) 

path = '/data/'
gefs_nca = 'gefs/NCA/gefs_merged.nc'
gefs_wcus = 'tensor/gefs_mos_daily_tensor_precip.pkl'
x_train, y_train, x_test, y_test = pickle.load(open(path+gefs_wcus, 'rb'))

# spatial average cm/day
pr_gefs = xr.open_dataset(path + gefs_nca).Total_precipitation
pr_gefs = xr.concat([pr_gefs.sel(fhour=hour).shift(time=i) for i,hour in enumerate(pr_gefs.fhour)],'fhour')
pr_gefs = pr_gefs.sel(time=time_mask(pr_gefs))
pr_true = np.concatenate((y_train, y_test))

leads = [d for d in range(14)]
quantiles = [0.50, 0.75, 0.80, 0.85, 0.9, 0.95]
# threshold = 0.5 # by default, but should be higher to reduce FP

# whole period. test period starts 2011.
results = pd.DataFrame(index=pd.date_range('1985-01-01','2019-12-31'))
results = results[is_NDJFM(results.index.month)]

# binary & continuous values at the very front

results['ERA5'] = pr_true

for quantile in quantiles:

  q = np.quantile(pr_true, quantile)
  results['ERA5_%0.2f' % quantile] = (pr_true > q)

  for l in leads:

    # print(quantile, l)

    fhours = pr_gefs.fhour[l] # get interval window
    prf = pr_gefs.sel(fhour=fhours)
    prf_q = np.quantile(prf.values, quantile)
    results['GEFS_VAL_%d' % l] = prf.values
    results['GEFS_%d_%0.2f' % (l, quantile)] = (prf.values > prf_q)

results.to_csv('/results/benchmark_matrix_precip_apples.csv')

# how many columns does the benchmark outcome have?
# 'ERA5'           1 column for true precip
# 'GEFS_VAL_%d'    14 columns for NCA precip mean at different lead time
# 'ERA5_%0.2f'     6 columns for true precip binary under different quantile
# 'GEFS_%d_%0.2f'  14 * 6 columns for NCA precip binary at lead & quantile

# reorganize benchmark matrix to get confusion matrix
binary_bm = results.copy()
for col in binary_bm.columns[1:]:
    if binary_bm[col].dtypes == 'float64':
        binary_bm[col] = (binary_bm[col]> 0.5)

leads = [d for d in range(14)]
quantiles = [0.50, 0.75, 0.80, 0.85, 0.9, 0.95]
con_precip_bm = pd.DataFrame(binary_bm['ERA5']) #'ERA5'

for q in quantiles:
    for l in leads:
        label = binary_bm['ERA5_%0.2f' % q] #'ERA5'
        pred =  binary_bm['GEFS_%d_%0.2f' % (l, q)]
        con_precip_bm['GEFS_%d_%0.2f_TP' % (l, q)] = (label&pred)
        con_precip_bm['GEFS_%d_%0.2f_TN' % (l, q)] = (~label&~pred)
        con_precip_bm['GEFS_%d_%0.2f_FP' % (l, q)] = (~label&pred)
        con_precip_bm['GEFS_%d_%0.2f_FN' % (l, q)] = (label&~pred)
        
con_precip_bm.to_csv('results/benchmark_confusion_matrix_apples.csv')

# how many columns does the benchmark confusion matrix have?
# 'ERA5'               1 column for true precip
# 'GEFS_%d_%0.2f_TP'   14 * 6 columns for NCA precip binary at lead & quantile
# 'GEFS_%d_%0.2f_TN'   14 * 6 columns for NCA precip binary at lead & quantile
# 'GEFS_%d_%0.2f_FP'   14 * 6 columns for NCA precip binary at lead & quantile
# 'GEFS_%d_%0.2f_FN'   14 * 6 columns for NCA precip binary at lead & quantile

