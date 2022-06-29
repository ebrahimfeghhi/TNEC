'''
The signal parameters are adopted from the YASA sleep spindle package by Rapheal Vallat.
'''

from mne.filter import filter_data
import numpy as np
import scipy
from scipy import signal
from scipy.signal import detrend
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy.random import default_rng
from scipy.signal import detrend
from model_architectures import convnetLFP
import pickle 
from numpy.random import default_rng
from sklearn.preprocessing import MinMaxScaler


class candidate_marker:
    
    def __init__(self, data, window_length, window_overlap,
                 broad_freq, freq_band, sampling_frequency, candidate_thresh,
                 rel_pow_band):
        
        self.data = data
        self.sf = sampling_frequency
        self.tr = sampling_frequency/1000
        self.window_length = int(window_length / self.tr)
        self.window_overlap = int(window_overlap / self.tr) 
        self.slide_length = window_length - window_overlap
        self.broad_freq = broad_freq
        self.freq_band = freq_band 
        self.candidate_thresh = candidate_thresh
        self.rel_pow_band = rel_pow_band
      
    def preprocess(self):
        
        '''applies band pass filter to remove frequences below and above 
        a certain cut-off
        '''
        
        self.data = filter_data(self.data, self.sf, 
                self.broad_freq[0], self.broad_freq[1], method='fir', verbose=0)
        
    def mark_candidates(self):

        self.window_data()

        '''detects candidate TNEs based on rel_pow, moving_corr, and moving_rms'''
        self.rel_pow()
        self.moving_corr()
        self.moving_rms()

        self.signal_params = np.vstack((self.rel_pow, self.moving_corr, self.moving_rms)).T
        self.remove_outliers()
        
        # convert features into percentiles 
        rel_quantile = scipy.stats.rankdata(self.signal_params[:, 0],
                                            "average")/self.num_windows
        corr_quantile = scipy.stats.rankdata(self.signal_params[:, 1],
                                             "average")/self.num_windows
        rms_quantile = scipy.stats.rankdata(self.signal_params[:, 2],
                                            "average")/self.num_windows
        self.quantiles = np.vstack((rel_quantile, corr_quantile, rms_quantile)).T

        # obtain inclusive scores by averaging over 3 percentiles 
        self.candidates = np.argwhere(np.average(self.quantiles, axis=1) > self.candidate_thresh)

        #self.check_params()
        
        # create candidate start stop times list 
        self.cand_start_stop = []
        for c in self.candidates:
            self.cand_start_stop.append((self.window_times[int(c)], 
            self.window_times[int(c)]+self.window_length))
        
        print("Out of",self.num_windows,"windows",self.candidates.shape[0],
              "candidates were detected")

    def window_data(self):
        
        '''
        windowed_raw: raw signal binned into windows (numpy array)
        windowed_filt: filtered signal binned into windows (numpy array)
        times: start time for each window (list)
        '''

        # starting at t=window_length, move forward slide_length
        # repeat until reach end of data, which is int(data_length/slide_length times)
        # add back 1 to include first window 
        self.num_windows = int((self.data.shape[0]-self.window_length)/(self.slide_length)) + 1

        # filter data
        self.filt_data = filter_data(self.data, self.sf, self.freq_band[0], self.freq_band[1], 
                                     l_trans_bandwidth=1.5,
                                     h_trans_bandwidth=1.5, method='fir', verbose=0)

        self.windowed_raw = np.zeros((self.num_windows, self.window_length))
        self.windowed_filt = np.zeros((self.num_windows, self.window_length))
        self.window_times = []
        
        for w in range(self.num_windows):
            w_start = w*self.slide_length
            w_end = w_start + self.window_length
            self.windowed_raw[w] = self.data[w_start:w_end]
            self.windowed_filt[w] = self.filt_data[w_start:w_end]
            self.window_times.append(w_start)

    def rel_pow(self):
        
        '''computes rel_pow in freq_band for every window
        This code is taken from Rapheal Vallat's YASA sleep spindle algorithm'''
        
        f, t, Zxx = signal.stft(self.data, self.sf, nperseg=self.window_length,
                                noverlap=self.window_overlap,
                               boundary=None, padded=False)

        # apply broadband filter
        idx_broad_band = np.logical_and(f >= self.rel_pow_band[0], f <= self.rel_pow_band[1])
        f = f[idx_broad_band]
        Zxx = Zxx[idx_broad_band, :]

        Zxx = np.square(np.abs(Zxx))

        # normalize 
        sum_pow = Zxx.sum(0).reshape(1, -1)
        np.divide(Zxx, sum_pow, out=Zxx)
         
        idx_freq_band = np.logical_and(f >= self.freq_band[0], f <= self.freq_band[1])
        self.rel_pow = Zxx[idx_freq_band].sum(0)  

    def moving_corr(self):
        
        '''computes pearson corr between detrended raw and filtered signal
        for every window'''
    
        self.moving_corr = np.zeros(self.num_windows)

        self.detrended_windows = detrend(self.windowed_raw, axis=1)

        for w in range(self.num_windows):
            r, _ = pearsonr(self.detrended_windows[w], self.windowed_filt[w])
            self.moving_corr[w] = r
            
    def moving_rms(self):
        
        '''computes RMS of filtered data for every window'''
        self.moving_rms = np.sqrt(np.mean(self.windowed_filt**2, axis=1))

    def remove_outliers(self, outlier_quantile=.99):

        # remove windows that in the top 1% of RMS values 
        rms_thresh = np.quantile(self.moving_rms, outlier_quantile) 
        rms_outlier_ind = np.argwhere(self.moving_rms > rms_thresh)
        self.signal_params = np.delete(self.signal_params, rms_outlier_ind, axis=0)
        self.windowed_raw = np.delete(self.windowed_raw, rms_outlier_ind, axis=0)
        self.windowed_filt = np.delete(self.windowed_filt, rms_outlier_ind, axis=0)
        
    def check_params(self):

        '''
        Check a given window at random, to see if signal 
        parameters were correctly computed. 
        '''

        rng = np.random.default_rng()
        indices_to_check = rng.choice(self.candidates, 1, replace=False)
        selected_detrend_raw = self.detrended_windows[indices_to_check].squeeze()
        selected_filt = self.windowed_filt[indices_to_check].squeeze()
        rms = np.sqrt(np.mean(selected_filt**2))
        p, _ = pearsonr(selected_detrend_raw, selected_filt)

        assert round(float(rms),4) == round(float(self.moving_rms[indices_to_check][0]),4), print("Error in RMS")

        assert round(float(p),4) == round(float(self.moving_corr[indices_to_check][0]),4), print("Error in corr")

        
    def save_events(self, TNE_start_stop_path, save_name, frac_required, train_ratio=1):

        events = np.hstack((self.windowed_raw[self.candidates], 
                                    self.windowed_filt[self.candidates])).swapaxes(1,2)

        if train_ratio != 1: 
            num_train_events = int(train_ratio * events.shape[0])

            np.save(save_name + '_X_train', events[:num_train_events])
            np.save(save_name + '_timestamps_train', self.cand_start_stop[:num_train_events])

            np.save(save_name + '_X_baseline', events[num_train_events:])
            np.save(save_name + '_timestamps_baseline', self.cand_start_stop[num_train_events:])

        else: 
            np.save(save_name + '_X', events)
            np.save(save_name + '_timestamps', self.cand_start_stop)

        if len(TNE_start_stop_path) != 0:

            TNE_start_stop = np.load(TNE_start_stop_path)

            y = self.create_labels(TNE_start_stop, frac_required)

            print("Number of consistent events: ", np.argwhere(y==1).shape[0])
            print("Number of inconsistent events: ", np.argwhere(y==0).shape[0])

            if train_ratio != 1:
                np.save(save_name + '_y_train', y[:num_train_events])
                np.save(save_name + '_y_test', y[num_train_events:])

            else:
                np.save(save_name + '_y', y) 

    def create_labels(self, TNE_start_stop, frac_required):
        
        # unroll start stop into a vector of indices where TNEs are present
        tne_times = []
        for t_s_s in TNE_start_stop:
            tne_times.append(np.arange(t_s_s[0], t_s_s[1]+1, 1))
        tne_times = np.hstack(tne_times)
        
        y = []

        # for each candidate, check amount it intersects with any of the TNEs
        # if amount > frac_required, consider candidate as consistent (1)
        # else it is inconsistent (0)
        for c in self.cand_start_stop:

            cand_inds = np.arange(c[0], c[1]+1, 1)
            overlap = np.intersect1d(cand_inds, tne_times).shape[0]

            if overlap > int(frac_required*self.window_length):
                y.append(1)
            else:
                y.append(0)
            
        return np.asarray(y)

class TNEC_models():
    
    def __init__(self, X_train, y_train, X_baseline, y_baseline, detrend_bool, 
                results_path, class_ratio=None):


        if detrend_bool: 
            X_train = self.detrend_func(X_train)
            X_baseline = self.detrend_func(X_baseline)

        if class_ratio != None:
            X_train, y_train = self.modify_class_ratio(X_train, y_train, class_ratio)
            X_baseline, y_baseline = self.modify_class_ratio(X_baseline, y_baseline, class_ratio)
            
        self.X_train = X_train
        self.y_train = y_train
        self.X_baseline = X_baseline
        self.y_baseline = y_baseline
        self.detrend_bool = detrend_bool
        self.results_path = results_path

    def detrend_func(self, X):
        
        X_raw_detrend = detrend(X[:, :, 0])
        X[:, :, 0] = X_raw_detrend 
        return X
            
    def modify_class_ratio(self, X, y, class_ratio, seed=0):

        '''
        @param class_ratio (float): specifies ratio of inconsistent to consistent events
        @param seed (int): seed used to initialize random number generate for selecting
        which inconsistent events to delete
        '''
        rng = np.random.default_rng(seed)
        inconsistent_indices = np.argwhere(y==0)
        num_consistent = np.argwhere(y==1).shape[0]

        num_inconsistent_delete = int(inconsistent_indices.shape[0] - (class_ratio * num_consistent))
        indices_delete = rng.choice(inconsistent_indices, num_inconsistent_delete, replace=False)

        return np.delete(X, indices_delete, axis=0), np.delete(y, indices_delete, axis=0)

    def model_specs(self, dropout, learning_rate, batch_size, dense_num, filters, 
                    kernel_size, nonlin_convblock, nonlin_output, 
                    loss_func, num_outputs):
        
        '''
        @param dropout: % of units zeroed out during training 
        @param learning_rate: step size for updating weights 
        @param batch_size: amount of examples to use for a single gradient update
        @param dense_num: number of dense units 
        @param optimizer: algorithm used to optimize weights
        @param filters: number of filters for CNN
        @param kernel_size: filter size 
        @param nonlin_convblock: nonlinearity used in conv block and first dense layer
        @param nonlin_output: nonlinearity used for dense output layer 
        @param loss_func: formula for computing loss
        @param num_outputs: number of output units
        '''
        self.size = self.X_train.shape[1]
        self.dims = self.X_train.shape[2]
        self.drop = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dense_num = dense_num
        self.filters = filters
        self.kernel_size = kernel_size
        self.nonlin = nonlin_convblock
        self.nonlin_out = nonlin_output 
        self.loss_func = loss_func
        self.num_outputs = num_outputs
        
        self.model_info = {'dropout': dropout, 'learning_rate': learning_rate, 
                            'batch_size': batch_size, 'dense_num': dense_num,
                             'filters': filters, 'kernel_size': kernel_size, 'nonlin_convblock': 
                           nonlin_convblock, 'nonlin_output':nonlin_output, 'loss_func': loss_func,
                           'num_outputs': num_outputs, 'detrend_bool':self.detrend_bool}
        
                    
    def perform_kfold(self, run_name, random_state=42, splits=5, epochs=20000, patience=600):
        
        '''
        @param run_name (str): name of folder to save kfold outputs
        @param random_state: ensures splts are constant across several k fold runs
        @param splits: integer indicating number of folds to split data
        @param epochs: number of epochs to run before terminating, set to a large value b/c 
                will most likely terminate early from early stopping 
        @param patience: if val loss does not improve after this many epochs, terminate training
        '''
        
        kfold_folder = self.results_path + run_name
        
        os.makedirs(kfold_folder, exist_ok=True)

        kf = StratifiedKFold(n_splits=splits, random_state=random_state, shuffle=True)
        train_loss = np.zeros(splits)
        val_loss = np.zeros(splits)
        num_epochs_list = np.zeros(splits)
        accuracy_list = np.zeros(splits)
        recall_list = np.zeros(splits)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        print("X SHAPE: ", self.X_train.shape)
        print("Y SHAPE: ", self.y_train.shape)
        
        for i, (train_index, val_index) in enumerate(kf.split(self.X_train, self.y_train)):

            X_train_k, X_val_k = self.X_train[train_index], self.X_train[val_index]
            y_train_k, y_val_k = self.y_train[train_index], self.y_train[val_index]
            
            model_k = convnetLFP(self.size, self.dims, self.filters, 
                    self.kernel_size, self.nonlin, self.nonlin_out, 
                    self.num_outputs, self.drop, self.dense_num)

            model_k.compile(loss=self.loss_func, metrics=['accuracy', 
                             tf.keras.metrics.Recall()],optimizer=opt)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
            mc = ModelCheckpoint(kfold_folder + '/best_model.h5', monitor='val_loss', mode='min', verbose=1,
                                 save_best_only=True)

            history = model_k.fit(X_train_k, y_train_k, batch_size=int(self.batch_size),
                                  epochs=epochs, 
                                  validation_data=(X_val_k, y_val_k), shuffle=False, 
                                  callbacks=[es, mc])

            model_best = tf.keras.models.load_model(kfold_folder + "/best_model.h5")
            best_metrics = model_best.evaluate(X_val_k, y_val_k)
            train_loss[i] = np.min(history.history['loss'])
            val_loss[i] = best_metrics[0]
            accuracy_list[i] = best_metrics[1]
            recall_list[i] = best_metrics[2]

            # if maximum epochs are reached, store unaltered value
            # otherwise subtract patience value
            if len(history.history['loss']) == epochs:
                num_epochs_list[i] = epochs
            else:
                num_epochs_list[i] = len(history.history['loss']) - patience
         
        np.save(kfold_folder + "/" + 'train_loss', train_loss)
        np.save(kfold_folder + "/" + 'val_loss', val_loss)
        np.save(kfold_folder + "/" + 'num_epochs', num_epochs_list)
        np.save(kfold_folder + "/" + 'accuracy_list', accuracy_list)
        np.save(kfold_folder + "/" + 'recall_list', recall_list)
        
        with open(kfold_folder + '/model_info.pkl', 'wb') as f:
            pickle.dump(self.model_info, f)
            
        self.best_epoch = np.average(num_epochs_list)
        self.accuracy_avg = np.average(accuracy_list)
        self.recall_avg = np.average(recall_list)
            
    def compute_TNEC(self, n, testing_data, save_model, epochs):
        
        '''
        @param n: number of models to train to compute a given NC score
        @param test_arr: datasets to use for computing NC scores
        @param save_models: set to true to save N models 
        @param epochs: num_epochs to use, if set to None uses best_epoch from kfold runs 
        '''
        
        if save_model: 
            model_folder = self.results_path + 'models'
            os.makedirs(model_folder, exist_ok=True)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        acc_storage = np.zeros(n)

        
        for i in range(n):

            test_preds_arr = []
            
            model = convnetLFP(self.size, self.dims, self.filters, 
                    self.kernel_size, self.nonlin, self.nonlin_out, 
                    self.num_outputs, self.drop, self.dense_num)

            model.compile(loss=self.loss_func, metrics=['accuracy'], optimizer=opt)


            history = model.fit(self.X_train, self.y_train, batch_size=int(self.batch_size),
                                epochs=epochs, shuffle=True,
                                verbose=1)
            
            #return model 
            if save_model:
                tf.keras.models.save_model(model, model_folder + '/model_' + str(i))
            
            acc_storage[i] = model.evaluate(self.X_baseline, self.y_baseline)[1]

            print("BASELINE ACCURACY: ", acc_storage[i])

            for t in testing_data:
                
                X_test = self.detrend_func(t)
                test_preds = model.predict(X_test)
                
                test_preds[test_preds >= .5] = 1
                test_preds[test_preds != 1] = 0
            
                test_preds_arr.append(test_preds)
                
            np.save(self.results_path + 'test_preds_' + str(i), np.stack(test_preds_arr))

        np.save(self.results_path + 'accuracy_baseline', acc_storage)