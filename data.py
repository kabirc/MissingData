import numpy as np
import scipy
from math import e
from collections import deque

class DataMatrix:
    """
    Abstract data type to store data matrices.  Different assumptions on data
    inherit from this class.
    """
    def __init__(self, data_dict):
        """
        data_dict contains keys 'n', 'p'
        """
        self.n = data_dict['n']
        self.p = data_dict['p']
        self.X = None 
        self.X_hat = None
        self.lam = None
        self.type = None

    def get_X(self):
        return self.X

    def get_X_hat(self, Z):
        return self.X_hat


class DiagonalData(DataMatrix):
    """
    Simplest example of data matrix.  Gaussian with diagonal covariance matrix.
    """
    def __init__(self, diag_dict):
        DataMatrix.__init__(self, diag_dict)
        self.cov = np.diag(diag_dict['diagonal'])
        self.diag = diag_dict['diagonal']
        # Data is generated in the class for this simple model
        self.X = np.random.multivariate_normal(np.zeros(self.p), self.cov, self.n)
        self.type = 'Identity'

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


class ARData(DataMatrix):
    """
    AR(1) data.  

    Takes in a parameter phi that sets the gain of the process.
    """
    def __init__(self, ar_dict, phi = 0.1):
        DataMatrix.__init__(self, ar_dict)
        self.approx = ar_dict['approx'] # true if need to approx cov
        self.alpha = ar_dict['alpha']
        self.phi = phi
        self.cov = self.create_covariance()
        self.X = np.random.multivariate_normal(np.zeros(self.p)\
                , self.cov, self.n)

    def create_covariance(self):
        cov = np.ones((self.p, self.p))
        cov_vals = self.phi**(np.array(range(self.p)))
        cov = 1/(1 - self.phi**2) * scipy.linalg.toeplitz(cov_vals)
        return cov


    def get_X_hat(self, Z):
        """
        Return the imputed matrix
        """
        X_hat = np.zeros(Z.shape)
        phi_hat = self.get_phi(Z)
        for i in range(self.n):
            Z_i = np.array(Z[i,:])
            S_i = np.where(~np.isnan(Z_i))[0] # Assumes this list is sorted
            prev = -1
            for k in S_i:
                # case 1: no observations to the left
                if prev == -1:
                    for a in range(k):
                        X_hat[i, a] = phi_hat**k * Z[i,k]
                    prev = k

                # case 2: Standard case (obs. on left and right)
                else:
                    for a in range(prev, k):
                        d1, d2 = a - prev, k - a # distance to left/right, resp.
                        prefactor = phi_hat**(d1+d2)\
                                    /(1 - phi_hat**(2 * (d1+d2)))
                        term_left = Z[i,prev] * (phi_hat**(-d2) - phi_hat**(d2))
                        term_right = Z[i,k] * (phi_hat**(-d1) - phi_hat**(d1))
                        X_hat[i,a] = prefactor * (term_left + term_right)
                    prev = k

            # case 3: no observations to the right
            for a in range(prev, self.p):
                X_hat[i,a] = phi_hat**(self.p - a) * Z[i,prev]

        self.X_hat = X_hat
        return X_hat


    def get_phi(self, Z):
        """
        Return the estimate of phi
        """
        if self.approx:
            num, denom = 0, 0 
            for i in range(self.n):
                for a in range(self.p - 1):
                    if np.isnan(Z[i,a]) or np.isnan(Z[i,a+1]):
                        num += 0
                    else: 
                        num += Z[i,a] * Z[i, a+1]
                    if np.isnan(Z[i,a]):
                        denom += 0
                    else: 
                        denom += Z[i,a]**2
            num = 1/(self.alpha **2 *self.n * (self.p - 1)) * num
            denom = 1/(self.alpha * self.n * (self.p - 1)) * denom
            return num/denom

        else: 
            return self.phi


class BandedData(DataMatrix):
    """
    Band Diagonal Data.  We use the matrix of example 4.1 in Wang, Wang,
    Balakrishnan, Singh.

    Creates a band diagonal precision matrix with entries phi^{|i - j|} where
    phi < 0.5.  Takes in a parameter n_bands that truncates if |i - j| >
    n_bands.  We default setting n_bands = 3.
    """
    def __init__(self, banded_dict, n_bands = 3, phi = 0.25):
        DataMatrix.__init__(self, banded_dict)
        self.approx = banded_dict['approx'] # true if need to approx cov
        self.alpha = banded_dict['alpha'] # nan if need to estimate alpha
        self.omega = self.create_precision_mat(n_bands, phi)
        # Returns adjacency list representation of graphical model
        self.graphical_model = self.get_graphical_model(self.omega)
        # The following matrix inversion could be made faster using sparse
        # library
        self.cov = np.linalg.inv(self.omega) 
        self.X = np.random.multivariate_normal(np.zeros(self.p), self.cov, self.n)
        self.type = 'Banded'

    def get_graphical_model(self, omega):
        """
        Creates the graphical model corresponding to the precision matrix passed
        in (omega).  The Gaussian model implies that zeros in the precision
        matrix correspond to absence of edge in the graph.  We store as an
        adjacency list for computational purposes as we will traverse the graph
        a lot
        """
        adj = (omega != 0).astype(int) # convert to adjacency matrix
        adj_list = {} 
        for i in xrange(self.p):
            adj_list[i] = np.where(adj[i] != 0)[0]    
        return adj_list

    def create_precision_mat(self, n_bands, phi):
        """
        Creates the precision matrix (faster than iterating through each entry)
        """
        omega = np.zeros((self.p, self.p))
        for i in range(n_bands):
            if i == 0:
                to_add = phi**i * np.diag(np.ones(self.p - i), i)
                omega += to_add
            else:
                to_add_upper = phi**i * np.diag(np.ones(self.p-i), i)
                to_add_lower = phi**i * np.diag(np.ones(self.p-i), -i)
                omega = omega + to_add_upper + to_add_lower
        return omega

    def get_X_hat(self, Z):
        """
        Give the least squares imputation for Gaussians
        """
        X_hat = np.zeros(Z.shape)
        if self.approx:
            temp_cov = self.get_approx_cov(Z)
            for i in range(self.n):
                Z_i = np.array(Z[i,:])
                X_hat[i] = self.gaussian_impute_row(Z_i, temp_cov)
        else:
            for i in range(self.n):
                Z_i = np.array(Z[i,:])
                X_hat[i] = self.gaussian_impute_row(Z_i, self.cov)
        self.X_hat = X_hat
        return X_hat

    def temp_get_X_hat(self, Z):
        X_hat = np.zeros(Z.shape)
        if self.approx:
            temp_cov = self.get_approx_cov(Z)
            for i in range(self.n):
                Z_i = np.array(Z[i,:])
                X_hat[i] = self.temp_gaussian_impute_row(Z_i, temp_cov)
        else:
            for i in range(self.n):
                Z_i = np.array(Z[i,:])
                X_hat[i] = self.temp_gaussian_impute_row(Z_i, self.cov)
        self.X_hat = X_hat
        return X_hat

    def get_zero_impute(self, Z):
        """ 
        Returns the zero imputed matrix
        """
        X_zero = np.zeros(Z.shape)
        for i in range(self.n):
            for j in range(self.p):
                if not np.isnan(Z[i,j]):
                    X_zero[i,j] = Z[i,j]
        return X_zero 

    def get_approx_cov(self, Z):
        """
        Approximate the covariance matrix 
        """
        approx_cov = np.zeros((self.p, self.p))
        if np.isnan(self.alpha):
            raise NotImplementedError
        else: 
            X_zero = self.get_zero_impute(Z)
            for i in range(self.n):
                X_zero_i = np.array(X_zero[i,:])
                outer_prod = np.outer(X_zero_i, X_zero_i) 
                outer_diag = np.diag(np.diag(outer_prod))
                approx_cov = approx_cov + 1/(self.alpha**2 * self.n) * outer_prod\
                        + (1 - self.alpha)/(self.alpha**2 * self.n) * outer_diag
        return approx_cov


    def temp_gaussian_impute_row(self, Z_i, cov):
        """
        Helper function that returns the row to impute and takes in the observed
        row and covariance (covariance can be approximated or true covariance)
        """
        to_impute = np.zeros(self.p)
        # Observed indices and entries in row
        S, X_S = np.where(~np.isnan(Z_i))[0], Z_i[~np.isnan(Z_i)] 
        Sc = np.where(np.isnan(Z_i))[0] # Missing indices 
        # Inverse covariance submatrix
        inv_cov_SS = np.linalg.inv(cov[np.ix_(S,S)])
        # Covariance matrix of unobserved-observed entries
        cov_ScS = cov[np.ix_(Sc, S)]
        X_hat_Sc = np.dot(np.dot(cov_ScS, inv_cov_SS), X_S)
        to_impute[S] = X_S
        to_impute[Sc] = X_hat_Sc
        return to_impute 

    def gaussian_impute_row(self, Z_i, cov):
        """
        Helper function that returns the row to impute.  Uses the bfs to only
        invert a small matrix
        """
        to_impute = np.zeros(self.p)
        # Observed indices and entries in row
        S, X_S = np.where(~np.isnan(Z_i))[0], Z_i[~np.isnan(Z_i)] 
        Sc = np.where(np.isnan(Z_i))[0] # Missing indices
        to_impute[S] = X_S
        Sc_impute = np.zeros(Sc.shape[0])

        # Want to avoid replicating work
        seen = {}
        for j, idx in zip(range(Sc.shape[0]), Sc):
            if idx in seen:
                Sc_impute[j] = seen[idx]
            else:
                blanket, equiv_nodes = self.find_blanket(idx, S)
                X_blanket = Z_i[blanket]
                inv_cov_blanket = np.linalg.inv(cov[np.ix_(blanket,blanket)])

                cov_idxS = cov[np.ix_(np.array([idx]), blanket)]
                idx_impute = np.dot(np.dot(cov_idxS, inv_cov_blanket), X_blanket)
                seen[idx] =  idx_impute
                Sc_impute[j] = idx_impute 
                for k in equiv_nodes:
                    cov_kS = cov[np.ix_(np.array([k]), blanket)]
                    seen[k] = np.dot(np.dot(cov_kS, inv_cov_blanket), X_blanket) 
        to_impute[Sc] = Sc_impute 
        return to_impute


    def find_blanket(self, node, S):
        """
        Find the blanket of node where S is the indices of the observed entries
        Thanks
        https://codereview.stackexchange.com/questions/135156/bfs-implementation-in-python-3
        for bfs implementation
        """
        visited, queue = set(), deque([node])
        blanket, nodes = [], [] # nodes is the nodes with same blanket
        while queue:
            vertex = queue.popleft()
            for neighbor in self.graphical_model[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if np.any(S == neighbor):
                        blanket.append(neighbor)
                    else:
                        nodes.append(neighbor)
                        queue.append(neighbor)
        return blanket, nodes 

class RealData(DataMatrix):
    """
    Performs imputation on real data.  Assumes data comes from a multivariate
    Gaussian.  Proceeds in two steps
    1. Use Graphical Lasso to find the graphical model
    2. From the approximate covariance matrix, impute values (covariance matrix
    found in the standard way assuming MCAR)
    """




