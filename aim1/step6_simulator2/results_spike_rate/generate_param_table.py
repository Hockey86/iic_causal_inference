import sys
import pickle
import numpy as np
import pandas as pd


#data_type = 'humanIIC'
data_type = 'CNNIIC'
#response_tostudy = 'iic_burden_smooth'
response_tostudy = 'spike_rate'

with open(f'../../data_to_fit_{data_type}_iic_burden_smooth+spike_rate.pickle', 'rb') as f:
    res = pickle.load(f)
for k in res:
    exec('%s = res[\'%s\']'%(k,k))
Pobs = Pobs[response_tostudy]
       
model = sys.argv[1]#cauchy_expit_lognormal_drugoutside_ARMA
AR_p = 2
MA_q = 6
maxiter = 1000
N = len(sids)
Ndrug = len(Dname)
with open(f'model_fit_{data_type}_{response_tostudy}_{model}{AR_p},{MA_q}_iter{maxiter}.pkl', 'rb') as ff:
    stan_model, df_params_, Ncluster = pickle.load(ff)

for posterior_stat in ['mean', 'std']:
    if posterior_stat == 'mean':
        df_params = df_params_.mean(axis=0)
    elif posterior_stat == 'std':
        df_params = df_params_.std(axis=0)

    data = {'SID':sids, 'cluster':cluster}

    # max D and max P
    data['maxD'] = [x.max() for x in D]
    data['maxE'] = [np.nanmax(x) for x in Pobs]

    # alpha0
    data['alpha0'] = [df_params['alpha0[%d]'%i] for i in range(1,N+1)]

    # alpha
    for p in range(1,AR_p+1):
        data['alpha[%d]'%p] = [df_params['alpha[%d,%d]'%(i,p)] for i in range(1,N+1)]

    # theta
    for q in range(1,MA_q+1):
        data['theta[%d]'%q] = [df_params['theta[%d,%d]'%(i,q)] for i in range(1,N+1)]
        
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
    df = df.rename(columns={'b[%d]'%(i+1,):'b[%s]'%Dname[i] for i in range(len(Dname))})
    df.to_csv(f'params_{posterior_stat}_{data_type}_{response_tostudy}_{model}{AR_p},{MA_q}_iter{maxiter}.csv', index=False)

