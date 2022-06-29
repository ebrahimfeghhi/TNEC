import numpy as np
from TNEC import candidate_marker
import mne
from scipy import stats
from automated_markings import automated_reviewers_paper
import pandas as pd
import yaml


# load yaml file 
with open('setup_data.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

data_path_R = config['data_path_R']
data_path_T = config['data_path_T']
sf = config['sf']
window_length = config['window_length']
window_overlap = config['window_overlap']
broad_freq = config['broad_freq']
freq_band = config['freq_band']
rel_pow_band = config['rel_pow_band']
candidate_threshold = config['candidate_threshold']
overlap_criteria = config['overlap_criteria']
TNEs_marked = config['TNEs_marked']
TNE_indices_path_R = config['TNE_indices_path_R']
TNE_indices_path_T = config['TNE_indices_path_T']
save_path_R= config['save_path_R']
save_path_T= config['save_path_T']

# init candidate_marker class, preprocess data, mark candidates, and save events
# for both R and T 
marker_R = candidate_marker(np.load(data_path_R), window_length, window_overlap,
                broad_freq, freq_band, sf, candidate_threshold, rel_pow_band) 
marker_R.preprocess()
marker_R.mark_candidates()
marker_R.save_events(TNE_indices_path_R, save_path_R, overlap_criteria, config['train_ratio'])

np.save('509_data/R_509_1_labels_new', marker_R.signal_params)
np.save('509_data/R_509_1_data_obi.npy', np.stack((marker_R.windowed_raw, marker_R.windowed_filt),axis=2))

marker_T = candidate_marker(np.load(data_path_T), window_length, window_overlap,
                broad_freq, freq_band, sf, candidate_threshold, rel_pow_band) 
marker_T.preprocess()
marker_T.mark_candidates()
marker_T.save_events(TNE_indices_path_T, save_path_T, overlap_criteria)


automated_reviewers_paper(marker_R.quantiles, marker_R.candidates, .9, 
                        'prepared_data/R_dataset_y', .8, False)
automated_reviewers_paper(marker_T.quantiles, marker_T.candidates, .9, 
                    'prepared_data/T_dataset_y', 1, True)



    

    
    
