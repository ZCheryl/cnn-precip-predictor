import numpy as np 
import pandas as pd
import pickle
from metrics import recall_np, precision_np, f1_np
from scipy.stats import norm

rname = '/results/benchmark_matrix_precip_apples.csv'
outcomes = pd.read_csv(rname, index_col=0, parse_dates=True)

def get_ci(x, conf_level): # half width
  Z = norm.ppf(0.5 + conf_level / 2)
  return Z*np.std(x, ddof=1)

max_leads = 14
quantiles = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
leads = [d for d in range(max_leads)]

results = {}
for quantile in quantiles:
  print(quantile)
  results[quantile] = {'precision': [], 'recall': [], 'f1': [], 
                       'precision_ci': [], 'recall_ci': [], 'f1_ci': []}

  for l in leads:
  
    y_true = outcomes['ERA5_%0.2f' % quantile]
    y_true = y_true[3932:] # y_true[3594:]  # only use test period
    y_pred = outcomes['GEFS_%d_%0.2f' % (l, quantile)]
    y_pred = y_pred[3932:] # [3594:]  # only use test period

    # bootstrap estimates
    N = y_true.size
    prec_r = []
    rec_r = []
    f1_r = []
    for i in range(100):
      r = np.random.randint(N, size=N)
      prec_r.append(precision_np(y_true[r], y_pred[r]))
      rec_r.append(recall_np(y_true[r], y_pred[r]))
      f1_r.append(f1_np(y_true[r], y_pred[r]))
    
    results[quantile]['precision'].append(np.mean(prec_r))
    results[quantile]['recall'].append(np.mean(rec_r))
    results[quantile]['f1'].append(np.mean(f1_r))
    
    results[quantile]['precision_ci'].append(get_ci(prec_r, 0.95))
    results[quantile]['recall_ci'].append(get_ci(rec_r, 0.95))
    results[quantile]['f1_ci'].append(get_ci(f1_r, 0.95))

    print(l, '%0.2f +/- %0.2f' % (np.mean(f1_r), get_ci(f1_r, 0.95)))

pickle.dump(results, open('/results/gefs_benchmark_precip_apples.pkl', 'wb'))
  