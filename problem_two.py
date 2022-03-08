from scipy.special import expit as invlogit
import numpy as np

def logistic_regression_IRLS(X, y, beta0, t):

    beta_t = np.zeros((t,X.shape[1]))
    beta_t[0] = beta0
    
    for i in range(1,t):
        pass #<complete>
        
    return beta_t[-1,:]