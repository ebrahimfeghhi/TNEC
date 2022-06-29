import numpy as np
from model_architectures import convnetLFP
import matplotlib
import os 
import tensorflow as tf
from TNEC import TNEC_models
import yaml

with open('compute_consistency_scores.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config['gpu']:
    os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_number']
    import tensorflow as tf
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth=True
    sess = tf.Session(config=config_tf)

kfold_bool = config['kfold_bool']
k = config['k']

X = np.load(config['R_data_path'])
y = np.load(config['R_labels_path'])

class_ratio = config['class_ratio']

detrend_bool = config['detrend_bool']

loss_func = config['loss_func']

# load model hyperparameters
drop = config['dropout']
learning_rate = config['learning_rate']
batch_size = config['batch_size']
kfold_epochs = config['kfold_epochs']
patience = config['patience']
dense_num = config['dense_num']
optim = config['optim']
filters = config['filters']
kernel_size = config['kernel_size']
nonlin = config['nonlin']
nonlin_out = config['nonlin_out']
loss_func = config['loss_func']
num_outputs = config['num_outputs']

save_path = config['save_path']

X_baseline = np.load(config['R_baseline_data_path'])
y_baseline = np.load(config['R_baseline_labels_path'])

TNEC = TNEC_models(X, y, X_baseline, y_baseline, detrend_bool, save_path, class_ratio)
TNEC.model_specs(drop, learning_rate, batch_size, dense_num, filters, 
             kernel_size, nonlin, nonlin_out, loss_func, num_outputs)

if kfold_bool: 
    TNEC.perform_kfold(config['run_name'], splits=k, patience=patience)
else:

    X_test = [] 
    for xt in config['T_data_path']:
        X_test.append(np.load(xt))

    TNEC.compute_TNEC(config['N'], X_test, True, config['epochs'])

    