import numpy as np
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import RMSprop
from metrics import f1_m
# from models_og import create_classifier
from models import create_classifier
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras

def get_class_weight(y):
#  return {0: y.sum() / y.size, 1: 1 - y.sum() / y.size}
   return {0: 0.5*y.size/(y.size-y.sum()), 1: 0.5*y.size/y.size}

    
def run_experiment(path = 'data/',
                   model_type = 'CNN',
                   lead = 1,
                   q = 0.95,
                   batch_size = 32,
                   epochs = 100,
                   dropout = None,
                   verbose = False):
    
    outfile = 'results/train_history_cnn/gefs_mos_%s_%0.2f_%d_%d' % (model_type, q, lead, epochs)
    fname = 'tensor/gefs_mos_daily_tensor_precip_center.pkl'
    
    x_train, y_train, x_test, y_test = pickle.load(open(path+fname, 'rb'))
    q_val = np.quantile(np.concatenate((y_train, y_test)), q)
    y_train = (y_train > q_val).astype(float)
    y_test = (y_test > q_val).astype(float)
      
    x_train = x_train[:,:,:,:,lead]
    x_test = x_test[:,:,:,:,lead]
    print('before oversampling', x_train.shape, y_train.shape)
    print('class weight', get_class_weight(y_train))


    # Plain Oversampling with imbleran
    ros = RandomOverSampler(random_state = 0)
    lat = x_train.shape[1]
    lon = x_train.shape[2]
    x_train, y_train = ros.fit_resample(x_train.reshape((-1, lat*lon)), y_train)
    x_train = x_train.reshape((-1, lat, lon, 1))

    print('after oversampling', x_train.shape, y_train.shape)
    # print(x_train[0,:,:,:])
    
    if model_type == 'CNN':
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
        
        lat_stacked = np.stack([lat_coord for _ in range(x_train.shape[0])],axis=0)
        lon_stacked = np.stack([lon_coord for _ in range(x_train.shape[0])],axis=0)
        x_train = np.stack((np.squeeze(x_train), lat_stacked, lon_stacked), axis=3)
        lat_stacked = lat_stacked[0:x_test.shape[0],:,:]
        lon_stacked = lon_stacked[0:x_test.shape[0],:,:]
        x_test = np.stack((np.squeeze(x_test), lat_stacked, lon_stacked), axis=3)


    input_shape = x_train.shape[1:] # channels last
    # print(input_shape)
    
    model = create_classifier(input_shape = input_shape,
                              type = model_type,
                              dropout = dropout,
                              nclass = 1)
    
    if verbose:
      print('Quantile: %0.2f\n lead: %d\n model: %s' % (q, lead, model_type))
      print('Training set: ~%0.2f%%' % ((y_train.sum() / y_train.size) * 100))
      print('Test set: ~%0.2f%%' % ((y_test.sum() / y_test.size) * 100))
      model.summary()
    
    if lead < 2:
        opt = keras.optimizers.Adam(learning_rate=0.0001)

    elif lead < 6:
        
        opt = keras.optimizers.Adam(learning_rate=0.00005)
        
    else:
        opt = keras.optimizers.Adam(learning_rate=0.00001)
    
    model.compile(loss='binary_crossentropy',
                  optimizer = opt,
                  metrics=[f1_m])
    
    # callback to save the best model seen during training
    mc = ModelCheckpoint(outfile+'_bestmodel.h5',
                          monitor='val_f1_m', mode='max', verbose=verbose, 
                          save_best_only=True)
    
    # patient early stopping callback
    es = EarlyStopping(monitor='val_f1_m', mode='max', 
                       verbose=verbose, patience=100)
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_test, y_test),
#                        class_weight = get_class_weight(y_train),
                        callbacks=[mc, es])
    
    history.model = None
    history = (history.history['loss'],
               history.history['f1_m'],
               history.history['val_loss'],
               history.history['val_f1_m'])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test F1:', score[1])
    print('Saved files:', outfile)
    pickle.dump(history, open(outfile+'_history.pkl', 'wb'))
