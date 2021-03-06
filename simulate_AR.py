from __future__ import division
from __future__ import print_function
import numpy as np
import csv
from sklearn.linear_model import Lasso
from random import randint
from collections import defaultdict
from imputation import ImputationMissingData
from data import DataMatrix, DiagonalData, BandedData, ARData
from math import e

if __name__ == '__main__':
    # Set the parameters
    n = 1000 
    p = 1200 
    sigma = 0 # additive noise
    sparse = True
    s = int(np.sqrt(p)) + 1 # square root sparsity

    if sparse:
        # Square root sparsity
        beta0 = np.zeros(p)
        for i in range(s):
            beta0[i] = 1/np.sqrt(s)
    else: 
        beta0 = np.ones(p)

    # Run a simulation
    num_trials = 20 
    alpha_list = np.linspace(0.7, 0.99, 10) # missingness parameter
    approx_error_vals, approx_max_error_vals, approx_min_error_vals = [], [], [] 
    error_vals, max_error_vals, min_error_vals = [], [], []
    for alpha in alpha_list:
        approx_avg_error, approx_max_error, approx_min_error = 0, 0, 100
        avg_error, max_error, min_error = 0, 0, 100
        for i in range(num_trials):
            ar_dict_approx = {'alpha': alpha\
                    , 'approx': True\
                    , 'n': n\
                    , 'p': p}

            ar_dict = {'alpha': alpha\
                    , 'approx': False\
                    , 'n': n\
                    , 'p': p}
            ar_gaussian_approx = ARData(ar_dict_approx)
            ar_gaussian = ARData(ar_dict)

            imputation_dict_approx = {'sparse': sparse, \
                    'beta0': beta0, \
                    'sigma': sigma, \
                    'data': ar_gaussian_approx, \
                    'alpha': alpha, \
                    'lambda': 1/alpha**4 * np.sqrt(np.log(p)/n)}

            imputation_dict = {'sparse': sparse, \
                    'beta0': beta0, \
                    'sigma': sigma, \
                    'data': ar_gaussian, \
                    'alpha': alpha, \
                    'lambda': 1/alpha**4 * np.sqrt(np.log(p)/n)}

            imputation_test_approx = ImputationMissingData(imputation_dict_approx)
            imputation_test = ImputationMissingData(imputation_dict)
            error_approx = imputation_test_approx.get_l2_error()
            error = imputation_test.get_l2_error()
            avg_error += error
            approx_avg_error += error_approx

            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
            if error_approx > approx_max_error:
                approx_max_error = error_approx
            if error_approx < approx_min_error:
                approx_min_error = error_approx

        avg_error = avg_error/num_trials
        approx_avg_error = approx_avg_error/num_trials
        error_vals.append(avg_error)
        max_error_vals.append(max_error)
        min_error_vals.append(min_error)
        approx_error_vals.append(approx_avg_error)
        approx_max_error_vals.append(approx_max_error)
        approx_min_error_vals.append(approx_min_error)

    ### Save the data
    data = np.array([alpha_list, error_vals, min_error_vals, max_error_vals, \
                 approx_error_vals, approx_min_error_vals, approx_max_error_vals]).transpose()
    print(data.shape)
    np.savetxt('ARGaussian.dat', data, fmt=['%.2f', '%.4f', '%.6f', '%.8f',\
            '%.10f', '%.12f', '%.14f'],\
            header='alpha   err min_err max_err apx_err apx_min_err apx_max_err', comments='# ')
