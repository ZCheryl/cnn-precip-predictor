import numpy as np 
# import pandas as pd
import xarray as xr
# import matplotlib.pyplot as plt
from keras.models import load_model
from metrics import f1_m
import os
from jh_saliency import visualize_saliency_jh_init, visualize_saliency_jh_run
# from vis.utils import utils
# from keras import activations
from keras.utils import CustomObjectScope
from tqdm import tqdm
# import os

np.random.seed(1337) # stochastic

def is_NDJFM(month):
  return (month >= 11) | (month <= 3)

def time_mask(ds):
  return (is_NDJFM(ds['time.month']) & 
         (ds['time.year'] >= 2011) & # test period
         (ds['time.year'] <= 2019)) 

path = 'data/'
gefs_file = 'gefs_combine/gefs_merged.nc'

pr_gefs = xr.open_dataset(path + gefs_file)
pr_gefs = pr_gefs.sel(time=time_mask(pr_gefs))
results = pr_gefs.copy().drop('Total_precipitation').drop_dims('fhour')
pr_gefs = pr_gefs.Total_precipitation

epochs = 50
leads = [d for d in range(14)]
models = ['MLP', 'CNN']#, 'MLP', 'VGG16', 'ConvLSTM']
quantiles = [0.50, 0.75, 0.80, 0.85, 0.9, 0.95]

# for some reason this takes a long time now (~hours)
for m in models:
  for quantile in quantiles:
    for l in leads:

      print(m,quantile,l)
      resultsfile = path+'results/explanation/exp_%s_%0.2f_%d.nc' % (m, quantile, l)
      if os.path.exists(resultsfile):
        print('skipping', resultsfile)
        continue

      saliency = np.zeros((1361,49,45))
      saliency_neg = np.zeros((1361,49,45))
      
      outfile = 'results/gefs/gefs_mos_%s_%0.2f_%d_%d' % (m, quantile, l, epochs)
      model = load_model(path+outfile+'_bestmodel.h5', 
                          custom_objects={'f1_m': f1_m})
      layer_idx = 9
#      if m == 'CNN':
#        layer_idx = utils.find_layer_idx(model, 'dense_494') # the last prediction layer
#      else:
#        layer_idx = utils.find_layer_idx(model, 'dense_137')
      fhour = pr_gefs.fhour[l]
      pr = pr_gefs.sel(fhour=fhour).values

      # initialize the saliency thing
      with CustomObjectScope({'f1_m': f1_m}):
        opt1 = visualize_saliency_jh_init(model, layer_idx, filter_indices=0) #, backprop_modifier='guided')
        opt2 = visualize_saliency_jh_init(model, layer_idx, filter_indices=0, grad_modifier='negate')
#, backprop_modifier='guided', grad_modifier='negate')

      for i in tqdm(range(pr_gefs.time.values.size)):
        
        X = np.expand_dims(pr[i], axis=2)

        # use grad_modifier='negate' to get the opposite
        saliency[i] = visualize_saliency_jh_run(model, opt1, layer_idx, filter_indices=0, 
                                                seed_input=X, backprop_modifier='guided')
        saliency_neg[i] = visualize_saliency_jh_run(model, opt2, layer_idx, filter_indices=0, 
                                                seed_input=X, backprop_modifier='guided', grad_modifier='negate')

      results['saliency'] = (['time', 'lat', 'lon'], saliency)
      results['saliency_neg'] = (['time', 'lat', 'lon'], saliency_neg)
      results.to_netcdf(resultsfile)
