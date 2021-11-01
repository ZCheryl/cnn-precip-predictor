# f1 scores 

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import os


path = 'data/'
models = ['CNN', 'MLP']#, 'MLP', 'VGG16', 'ConvLSTM']
leads = [d for d in range(14)]
quantiles = [0.50, 0.75, 0.80, 0.85, 0.9, 0.95]
threshold = 0.5
gefs_benchmark = pickle.load(open('results/gefs_benchmark_uniq.pkl', 'rb'))

if os.path.exists('results/f1scores_uniq.pkl'):
    results = pickle.load(open('results/f1scores_uniq.pkl', 'rb'))
else:
    results = {}
    rname = 'results/classifier_outcomes_uniq.csv'
    outcomes = pd.read_csv(rname, index_col=0, parse_dates=True)

    
    for q in quantiles:
        results[q] = {}
        y = outcomes['ERA5_%0.2f' % q]
        y_test = y[3594:] # test only for benchmark
        
        for m in models:
            f1model = []
            f1modelerr = []
            no_skill = []
            
            for l in leads:
                col = '%s_%d_%0.2f' % (m, l, q)
                y_pred = outcomes[col]
                y_pred = y_pred[3594:] # test period only for benchmark
                no_skill.append(y_test.sum() / y_test.size)
                
                # bootstrap
                nboot = 100
                temp = np.zeros(nboot)
                N = y_test.size
                
                for i in range(nboot):
                    r = np.random.randint(N, size=N)
                    temp[i] = f1_score(y_test[r], (y_pred[r] > threshold))

                f1model.append(np.mean(temp))
                f1modelerr.append(1.96 * np.std(temp, ddof=1)) 
                # how was the error calculated??

            results[q][m] = (f1model, f1modelerr)
            print(f1model)

    pickle.dump(results, open('results/f1scores_uniq.pkl', 'wb'))  