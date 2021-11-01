#from keras_experiment import run_experiment
from keras_experiment_1d_precip import run_experiment


model = 'MLP'
q = 0.8
batch_size = 32
epochs = 100
dropout = 0.2
# lead = 'xs' 
lead = 5

print(model, q, epochs,dropout, 'batch size', batch_size)

run_experiment(path = 'data/',
               model_type = model,
               lead = lead, # 0-13 index for 1-14 day leads
               q = q,
               batch_size = batch_size,
               epochs = epochs,
               dropout = dropout,
               verbose = 1)


# run_experiment(path = '/data/',
#                model_type = 'CNN',
#                lead = 9, # 0-13 index for 1-14 day leads
#                q = 0.9,
#                batch_size = 32,
#                epochs = 50,
#                dropout = 0.4,
#                verbose = True)