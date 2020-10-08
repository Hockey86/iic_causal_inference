import pickle
import numpy as np
import pandas as pd


with open('../data_to_fit.pickle', 'rb') as f:
    res = pickle.load(f)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
       
model = 'cauchy_expit_ARMA'       
AR_p = 1
MA_q = 6
maxiter = 1000   
N = len(sids)
Ndrug = len(Dname)
with open('model_fit_%s%d%d_iter%d.pkl'%(model, AR_p, MA_q, maxiter), 'rb') as ff:
    stan_model, fit_res = pickle.load(ff)
df_params = fit_res.to_dataframe(diagnostics=False)
df_params = df_params.drop(columns=['chain', 'draw', 'warmup', 'lp__'])
df_params = df_params.mean(axis=0)  # take posterior mean of parameters

data = {'SID':sids, 'cluster':cluster}

# alpha0
data['alpha0'] = [df_params['alpha0[%d]'%i] for i in range(1,N+1)]

# alpha
for p in range(1,AR_p+1):
    data['alpha[%d]'%p] = [df_params['alpha[%d,%d]'%(i,p)] for i in range(1,N+1)]

# theta
for q in range(1,MA_q+1):
    data['theta[%d]'%q] = [df_params['theta[%d]'%q]]*N
    
# sigma_err
data['sigma_err'] = [df_params['sigma_err[%d]'%(cluster[i]+1,)] for i in range(N)]

# b
for d in range(1,Ndrug+1):
    data['b[%d]'%d] = [df_params['b[%d,%d]'%(i,d)] for i in range(1,N+1)]

# for patients with some drug, make b NaN for the missing drug
for i in range(N):
    Dmax = D[i].max(axis=0)
    for d in range(Ndrug):
        if Dmax[d]==0:
            data['b[%d]'%(d+1,)][i] = np.nan

# save
df = pd.DataFrame(data=data)
df_C = pd.DataFrame(data=C, columns=Cname)
df = pd.concat([df, df_C], axis=1)
df.to_csv('params_%s%d%d_iter%d.csv'%(model, AR_p, MA_q, maxiter), index=False)

