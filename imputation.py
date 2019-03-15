import numpy as np
import cvxpy as cp

class ImputationMissingData:

    def __init__(self, input_dict, missingness='MCAR'):
        self.beta0 = input_dict['beta0']
        self.s = np.sum([1 for i in self.beta0 if i != 0]) # sparsity of beta0
        self.R = np.sqrt(np.sum([i**2 for i in self.beta0])) # radius of beta0
        self.sigma = input_dict['sigma'] # noise std
        self.data = input_dict['data'] # DataMatrix object 
        self.missingness = missingness 
        if self.missingness == 'MCAR':
            self.alpha = input_dict['alpha'] 
        self.X = self.data.get_X() # passed in data matrix
        self.n = self.X.shape[0] # Number of samples
        self.p = self.X.shape[1] # dimension of data

    def generate_data(self):
        """
        passes the data matrix X through the channel specified by missingness
        output: (y, Z) where y is the response vector, Z is the observed matrix
        """
        # First generate the observation matrix
        Z = np.zeros(self.X.shape) 
        if self.missingness == 'MCAR':
            for i in xrange(self.n):
                for j in xrange(self.p):
                    # Pass data through a BEC(alpha)
                    if np.random.random() <= self.alpha:
                        Z[i,j] = self.X[i,j]
                    else:
                        Z[i,j] = None # This becomes nan!
                    
        elif self.missingness == 'MAR':
            # This functionality has not yet been added
            raise NotImplementedError
        elif self.missingness == 'MNAR': 
            # This functionality as not yet been added
            raise NotImplementedError

        # Now generate the response vector
        w = np.random.normal(0, self.sigma, self.n) # noise
        y = np.dot(self.X, self.beta0) + w
        return {'y': y, 'Z': Z}

    def regress(self, X_hat, y):
        """
        Calls the appropriate regression formulation depending on low or high
        dimensional regime
        """
        # Used in both regimes
        def loss_fn(X, Y, beta):
            return 1/(2*self.n) * cp.pnorm(cp.matmul(X, beta) - Y, 2)**2

        beta = cp.Variable(self.p)
        problem = None
        if self.n > self.p:
            # In low dimensions, take the least squares solution
            problem = cp.Problem(cp.Minimize(loss_fn(X_hat, y, beta)))
        else:
            # Run the lasso
            lam = self.get_regularizer()
            # The following section follows the cvxpy lasso implementation from 
            # cvxpy: https://www.cvxpy.org/examples/machine_learning/lasso_regression.html 
            def regularizer(beta): 
                return cp.norm1(beta)

            def objective_fn(X, Y, beta, lambd):
                return loss_fn(X, Y, beta) + \
                        lambd * regularizer(beta)
            problem = cp.Problem(cp.Minimize(objective_fn(X_hat, y, beta, lam)))
        problem.solve()
        beta_hat = beta.value
        return beta_hat

    def get_regularizer(self):
        lam = None
        if self.missingness == 'MCAR':
            subgaussian_param = self.data.get_subgaussian_constant()
            lam = subgaussian_param * (self.sigma + np.sqrt(1 - self.alpha) * self.R) \
                    * np.sqrt(np.log(self.p)/self.n)
        elif self.missingness == 'MAR':
            raise NotImplementedError
        elif self.missingness == 'MNAR':
            raise NotImplementedError
        return lam

    def simulate(self):
        """
        Runs one trial of regression with missing data
        """
        data_dict = self.generate_data()
        y, Z = data_dict['y'], data_dict['Z']
        X_hat = self.data.get_X_hat(Z)
        beta_hat = self.regress(X_hat, y)
        return beta_hat

    def get_l2_error(self):
        """ 
        returns l2 error between beta0, beta_hat
        """
        beta_hat = self.simulate()
        return np.sqrt(np.sum([i**2 for i in (beta_hat - self.beta0)]))
