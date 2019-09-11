from __future__ import division
from __future__ import print_function
import numpy as np
import csv
from sklearn.linear_model import Lasso
from random import randint
from collections import defaultdict
from imputation import ImputationMissingData
from data import DataMatrix, DiagonalData
from math import e

if __name__ == '__main__':
    # Set the parameters
    # Set the parameters
    n = 100 
    p = 20 
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
    num_trials = 1 
    alpha_list = np.linspace(0.5, 0.99, 5)
    error_vals, max_error_vals, min_error_vals = [], [], [] 
    for alpha in alpha_list:
        avg_error, max_error, min_error = 0, 0, 10000
        for i in range(num_trials):
            diag_dict = {'diagonal': np.ones(p)\
                        , 'n': n\
                        , 'p': p}
            simple_gaussian = DiagonalData(diag_dict)

            imputation_dict = {'sparse': sparse\
                        , 'beta0': beta0\
                        , 'sigma': sigma\
                        , 'data': simple_gaussian\
                        , 'alpha': alpha\
                        , 'lambda': np.sqrt(alpha * (1 - alpha)) * np.sqrt(np.log(p)/n)}
            imputation_test = ImputationMissingData(imputation_dict)

            error = imputation_test.get_l2_error()
            avg_error += error
            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        avg_error = avg_error/num_trials

        error_vals.append(avg_error)
        max_error_vals.append(max_error)
        min_error_vals.append(min_error)

    data = np.array([alpha_list, error_vals, min_error_vals,max_error_vals]).transpose()
    print(data.shape)

    np.savetxt('IdentityGaussianBig.dat', data, \
            fmt=['%.2f', '%.4f', '%.6f', '%.8f'], \
            header='alpha   err min_err max_err', \
            comments='# ')
