import numpy as np
from scipy.stats import cauchy, binom
from scipy.special import expit as sigmoid
from scipy.special import logit
from tqdm import tqdm


def predict(D, A_start, cluster, t0, sigma0, alpha0, alpha, theta, sigma_err, b, W, AR_p, MA_q, random_state=None, verbose=True):
    np.random.seed(random_state)
    T, N, ND = D.shape
    N_sample = t0.shape[0]
    Tstart = A_start.shape[1]
    A = np.zeros((N_sample, N, T))+np.nan
    P_output = np.zeros((N_sample, N, T))+np.nan
    
    A[:,:,:Tstart] = A_start
    for n in tqdm(range(N_sample), disable=not verbose):
        err = cauchy.rvs(loc=0, scale=np.tile(sigma_err[n][cluster], (T,1)).T)
        err = err.reshape(N,-1)
        
        for t in range(Tstart, T):
            # AR(p)
            alpha0 = logit(np.exp( -(np.log(t-t0[n]))**2 / (2*sigma0[n]**2) ))
            A[n][:,t] = alpha0 + (alpha[n] * A[n][:,t-1::-1][:,:AR_p]).sum(axis=1)
            
            # MA(q)
            MA_q2 = min(MA_q, t)
            A[n][:,t] += (theta[n][:,:MA_q2] * err[:,t-1::-1][:,:MA_q2]).sum(axis=1)

        # drug
        P_output[n][:,AR_p:T] = A[n][:,AR_p:T] - (D[AR_p-1:T-1] * b[n]).sum(axis=-1).T

        # sample P_output
        prob = sigmoid(P_output[n])
        P_output[n] = binom.rvs(W, prob) / W
        
    return P_output, A
