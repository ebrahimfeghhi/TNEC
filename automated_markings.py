import numpy as np
from scipy import stats 
import pandas as pd

    
def automated_reviewers_paper(quantiles, cand_ind, thresh, save_path, 
                            train_ratio):
    
    '''
    marks TNEs according to automatic reviewers outlined in paper. 
    ''' 

    quantiles = quantiles[cand_ind.squeeze()]
    A_TNEs  = np.zeros(quantiles.shape[0])
    B_TNEs = np.zeros(quantiles.shape[0])
    C_TNEs = np.zeros(quantiles.shape[0])
    num_train = int(quantiles.shape[0] * train_ratio)
    
    # compute reviewer scores 
    A = np.average(quantiles, weights=[1/5., 3/5., 1/5.], axis=1)
    B = np.average(quantiles, weights=[3/5., 1/5., 1/5.], axis=1)
    C = np.average(quantiles, weights=[3/4., 1/8., 1/8.], axis=1)
    
    A_TNEs[np.argwhere(A > thresh)] = 1
    B_TNEs[np.argwhere(B > thresh)] = 1
    C_TNEs[np.argwhere(C > thresh)] = 1

    if train_ratio != 1:

        A_TNEs_train = A_TNEs[:num_train]
        A_TNEs_test = A_TNEs[num_train:]
        B_TNEs_train = B_TNEs[:num_train]
        B_TNEs_test = B_TNEs[num_train:]
        C_TNEs_train = C_TNEs[:num_train]
        C_TNEs_test = C_TNEs[num_train:]

        np.save(save_path + '_train_A', A_TNEs_train)
        np.save(save_path + '_train_B', B_TNEs_train)
        np.save(save_path + '_train_C', C_TNEs_train)

        np.save(save_path + '_baseline_A', A_TNEs_test)
        np.save(save_path + '_baseline_B', B_TNEs_test)
        np.save(save_path + '_baseline_C', C_TNEs_test)

    else:
        np.save(save_path + '_A', A_TNEs)
        np.save(save_path + '_B', B_TNEs)
        np.save(save_path + '_C', C_TNEs)
