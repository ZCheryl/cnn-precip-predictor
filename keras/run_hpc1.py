from keras_experiment_uniq import run_experiment
import sys

# submit a bunch of jobs on hpc1
# (also need loop over random trials?)

# initial: 30 things
max_lead = 14
models = ['CNN', 'MLP'] #, 'VGG16', 'ConvLSTM']
leads = [d for d in range(max_lead)]
quantiles = [0.5, 0.75, 0.8, 0.85, 0.9, 0.95]

settings = [(m,l,q) for m in models for l in leads for q in quantiles]

m,l,q = settings[int(sys.argv[1])]
print(m,l,q)

dropout = 0.4 if m == 'MLP' else 0.2

run_experiment(path = 'data/',
               model_type = m,
               lead = l,
               q = q,
               batch_size = 32,
               epochs = 50,
               dropout = dropout,
               verbose = True)
