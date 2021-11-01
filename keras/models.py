from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D

def create_classifier(type = 'MLP',
                      input_shape = None,
                      dropout = None,
                      nclass = 1):
  
  model = Sequential()

  if type == 'MLP':
      model.add(Flatten(input_shape=input_shape))

      model.add(Dense(256, activation='relu'))
      if dropout: model.add(Dropout(dropout))

      model.add(Dense(128, activation='relu'))
      if dropout: model.add(Dropout(dropout))

      model.add(Dense(64, activation='relu'))   
      model.add(Dense(8, activation='relu'))   

      model.add(Dense(nclass, activation='sigmoid'))


  elif type == 'CNN':
      model.add(Conv2D(32, kernel_size=(4,4), activation='relu', input_shape=input_shape))
      model.add(BatchNormalization())
 
      model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2, 2)))

      model.add(Conv2D(16, kernel_size=(2,2), activation='relu'))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(pool_size=(2, 2)))
    
      model.add(Flatten())
      model.add(Dense(64, activation='relu'))
      model.add(Dropout(0.2))
      
      model.add(Dense(16, activation='relu'))
      model.add(Dense(nclass, activation='sigmoid'))


  else:
    raise ValueError('Not a valid model type.')

  return model
