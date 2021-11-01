import numpy as np 
import pandas as pd
import pickle
from metrics import f1_m
from tensorflow.keras.models import load_model

def is_NDJFM(month):
  return (month >= 11) | (month <= 3)

path = 'data/'
gefs_wcus = 'tensor/gefs_mos_daily_tensor_precip_center.pkl'
x_train, y_train, x_test, y_test = pickle.load(open(path+gefs_wcus, 'rb'))
X = np.concatenate((x_train, x_test))
pr_true = np.concatenate((y_train, y_test))

epochs = 100
models = ['CNN', 'MLP']#, 'MLP', 'VGG16', 'ConvLSTM']
leads = [d for d in range(14)]
quantiles = [0.50, 0.75, 0.9, 0.95]# 0.80, 0.85, 0.9, 0.95]

# quantiles = [0.50, 0.75, 0 0.9, 0.95]
# threshold = 0.5 # by default, but should be higher to reduce FP

# whole period. test period starts 2011.
results = pd.DataFrame(index=pd.date_range('1985-01-01','2019-12-31'))
results = results[is_NDJFM(results.index.month)]

results['ERA5'] = pr_true


# create lat coord
lat_coord = np.linspace(14,62,49)-38
lat_coord[29:] = lat_coord[29:]-4
lat_coord[:24] = np.round(lat_coord[:24]/max(abs(lat_coord)),2)
lat_coord[29:] = np.round(lat_coord[29:]/max(abs(lat_coord)),2)
lat_coord[24:29] = 0

# create lon coord
# 97 - 141
# 263 - 219
# 240 - 237
lon_coord = np.linspace(219,263,45)-237
lon_coord[22:]=lon_coord[22:]-3
lon_coord[:18] = np.round(lon_coord[:18]/max(abs(lon_coord)),2)
lon_coord[22:] = np.round(lon_coord[22:]/max(abs(lon_coord)),2)
lon_coord[18:22] = 0
lon_coord = -lon_coord

# end result should be a rectangle matrix of the size 49*45
lat_coord = np.repeat(lat_coord,45).reshape(49,45)
lon_coord = np.transpose(np.repeat(lon_coord,49).reshape(45,49))

# match size of input to cnn (3932,49,45,3)

lat_stacked = np.stack([lat_coord for _ in range(X.shape[0])],axis=0)
lon_stacked = np.stack([lon_coord for _ in range(X.shape[0])],axis=0)


for quantile in quantiles:
  q = np.quantile(pr_true, quantile)
  results['ERA5_%0.2f' % quantile] = (pr_true > q)

  for l in leads:
        
    for m in models:
        if m == "CNN":
            input_X = np.stack((np.squeeze(X[:,:,:,:,l]), lat_stacked,lon_stacked), axis=3)
            epochs = 200
        else:
            input_X = X[:,:,:,:,l]
            epochs = 300
        
        col = '%s_%d_%0.2f' % (m, l, quantile)
        print(col)
        outfile = 'results/train_history_1d_precip_location_v9/gefs_mos_%s_%0.2f_%d_%d' % (m, quantile, l, epochs)
        
        model = load_model(outfile+'_bestmodel.h5', custom_objects={'f1_m': f1_m})
        
        # get input features from gefs west coast, and run prediction
        y_pred = model.predict(input_X)
      
        # class probability 0-1 of event > threshold
        results[col] = y_pred # (y_pred > threshold) 

results.to_csv('results/classifier_outcomes_1d_precip_location_v9.csv')

# how many columns does the classifier outcome have?
# 'ERA5'           1 column for true precip
# 'ERA5_%0.2f'     6 columns for true precip binary under different quantile
# '%s_%d_%0.2f'    14 * 6 columns for CNN softmax
# '%s_%d_%0.2f'    14 * 6 columns for MLP softmax
