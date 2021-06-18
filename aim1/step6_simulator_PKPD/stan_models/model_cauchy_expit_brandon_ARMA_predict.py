import numpy as np
from scipy.stats import cauchy, binom
from scipy.special import expit as sigmoid
from scipy.special import logit
from tqdm import tqdm


def predict(model_params, random_state=None, verbose=True):
    np.random.seed(random_state)
    Ddose = model_params['Ddose']
    cluster = model_params['cluster']
    t0 = model_params['t0']
    sigma0 = model_params['sigma0']
    lambda0 = model_params['lambda0']
    alpha0 = model_params['alpha0']
    alpha1 = model_params['alpha1']
    alpha2 = model_params['alpha2']
    halflife = model_params['halflife']
    gamma = np.exp(np.log(0.5)/halflife)
    beta = model_params['beta']
    sigma_err = model_params['sigma_err']
    W = model_params['W']

    T, N, ND = Ddose.shape
    N_sample = t0.shape[0]
    D = np.zeros((N_sample, N, T, ND))+np.nan
    A = np.zeros((N_sample, N, T))+np.nan
    P_output = np.zeros((N_sample, N, T))+np.nan
    
    if 'Astart' in model_params:
        Astart = model_params['Astart']
        Tstart = Astart.shape[0]
        A[:,:,:Tstart] = Astart
    else:
        Tstart = 0 
    if 'Dstart' in model_params:
        Dstart = model_params['Dstart']
        D[:,:,:Tstart] = Dstart

    for n in tqdm(range(N_sample), disable=not verbose):
        err = cauchy.rvs(loc=0, scale=np.tile(sigma_err[n][cluster], (T,1)).T)
        err = err.reshape(N,-1)

        for t in range(Tstart, T):
            # AR(p)
            if t==0:
                A[n][:,t] =  alpha0[n] + err[:,t]
            elif t==1:
                A[n][:,t] =  alpha0[n] + alpha1[n]*A[n][:,t-1] + err[:,t]
            elif t>=2:
                A[n][:,t] =  alpha0[n] + alpha1[n]*A[n][:,t-1] + alpha2[n]*A[n][:,t-2] + err[:,t]

            # drug concentration from PK
            if t==0:
                D[n][:,t] = Ddose[t]
            else:
                D[n][:,t] = gamma[n] * D[n][:,t-1] + Ddose[t]
            
        # sample P_output

        lognormal = (lambda0[n]*np.exp(-(np.log(np.arange(T).reshape(-1,1)-t0[n]))**2/(2**sigma0[n]*2))).T
        P_output[n][:,0] = sigmoid(lognormal[:,0]+A[n][:,0])
        P_output[n][:,1:T] = sigmoid(lognormal[:,1:T]+A[n][:,1:T]) * sigmoid(-(beta[n].reshape(N,1,ND)*D[n][:,:T-1]).sum(axis=-1))*2
        P_output[n] = binom.rvs(W, P_output[n]) / W
        
    return P_output, {'A':A,'D':D}

