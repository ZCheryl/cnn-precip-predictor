## cnn-precip-predictor

![fig 3](https://user-images.githubusercontent.com/17866544/139566152-78f998df-b3f7-4ca2-b259-9068619cf8c6.png)


This project aims to improve preiciptation forecast from GEFS with deep learning methods. With the capability of connecting spatial patterns to abstract concepts, CNN could potentially provide improvement to medium range regional forecast from the forecast of a larger spatial region. 

**Data** obtained from (https://psl.noaa.gov/forecasts/reforecast2/download.html). GEFS went through an update from v2 to v12 in Sept. 2020. `GEFS` folder contains preprocesing files. 
`data/gefs-merge-two-files.py` aggregates GEFS forecast to daily forecasts. 
`data/to_tensor.py` converts data to tensors input. In addition to the preicipitation layer, two more layers of lon/lat features are included. 

**Experiment** contains CNN and MLP models to generate probablistic predictions for target region. Run `run_local.py` to complete training of one model. Jupyter notebook `` helps monitor training quality. Run `` on a computing cluster to train models in batch. 

**Benchmark** statiscally process GEFS and creates binary predictions. Two benchmarks are available. The naive GEFS benchmark makes prediction by comparing the forecasted spatial average with the actual precipitation threshhold. The bias-corrected GEFS makes prediction essentially from ranking the forecasted spatial average.

**Analysis** evaluates model results with F1 score and ROC, comparing with the benchmarks. 

**Results** gathers all .csv files resulted from the experiment, benchmark and analysis.

  - `classifier_outcome` contains probabilistic predictions derived from model (1985-2019)

  - `confusion_matrix` contains binary results from models and benchmarks

  - `8_outcomes` contains 2*2*2 contigency table comparing ERA5 ground truth, bias-corrected GEFS and model results.

**Saliency** allows daily/categorical investigation on which input "pixels" contribute to a positive/negative classification.

**Figures** contains codes to generate all figures in the figures.ppt, including visualization of analysis, saliency, reliability diagram, specific examples, etc. Jupyter notebook `` helps visualize filters and features maps.

Package requirements: xarray, keras, cartopy
Tensorflow version: 2.3.0
Python 3.7.10

Reference: https://github.com/jdherman/cnn-streamflow-forecast
