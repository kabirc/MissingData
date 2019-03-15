import numpy as np

class DataMatrix:
    """
    Abstract data type to store data matrices.  Different assumptions on data
    inherit from this class.
    """
    def __init__(self):
        self.X = None 
        self.X_hat = None
        self.lam = None

    def get_X(self):
        return self.X

    def get_X_hat(self, Z):
        return self.X_hat


class DiagonalData(DataMatrix):
    """
    Simplest example of data matrix.  Gaussian with diagonal covariance matrix.
    """
    def __init__(self, diag_dict):
        self.cov = np.diag(diag_dict['diagonal'])
        self.diag = diag_dict['diagonal']
        self.n = diag_dict['n']
        self.p = diag_dict['p']
        # Data is generated in the class for this simple model
        self.X = np.random.multivariate_normal(np.zeros(self.p), self.cov, self.n)

    def get_X_hat(self, Z):
        # All that is necessary is 0 imputation
        X_hat = np.zeros(Z.shape)
        for i in range(self.n):
            for j in range(self.p):
                if not np.isnan(Z[i,j]):
                    X_hat[i,j] = Z[i,j]
        self.X_hat = X_hat
        return self.X_hat

    def get_subgaussian_constant(self):
        return np.max(self.diag)


