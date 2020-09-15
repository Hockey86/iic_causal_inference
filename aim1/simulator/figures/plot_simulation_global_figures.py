import os
import pickle
import numpy as np
from scipy.special import expit as sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt


W = 300
Dnames = ['lacosamide', 'levetiracetam', 'midazolam', 
          #'pentobarbital','phenobarbital',# 'phenytoin',
          'propofol', 'valproate']
ND = len(Dnames)
#models = ['lognormal']#, 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
models = ['ARMA16']

"""
with open('../results/model_fit_lognormal.pkl', 'rb') as f:
    stan_model, fit_res = pickle.load(f)   
N = 82 
df = fit_res.to_dataframe(pars=['alpha', 'mu', 'sigma'])#'t0', 
#t0    = np.array([df['t0[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
alpha = np.array([df['alpha[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
mu    = np.array([df['mu[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
sigma = np.array([df['sigma[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
"""

    
for model in models:
    print(model)
    
    figure_dir = 'simulation_global_figures/%s'%model
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    with open('../results/results_%s.pickle'%model, 'rb') as ff:
        res  = pickle.load(ff)

    Ep = res['Ep_sim']
    E = res['E']
    Dscaled = res['Dscaled']
    Dmax = res['Dmax']
    sids = res['sids']
    
    for si, sid in enumerate(tqdm(sids)):
        tt = np.arange(len(E[si]))*W*2/3600
        P = np.array(E[si]).astype(float)
        P[P==-1] = np.nan
        P = P/W*100
        
        plt.close()
        fig = plt.figure(figsize=(9,6))
        
        ax1 = fig.add_subplot(211)
        random_ids = np.random.choice(len(Ep[si]), 1, replace=False)
        ax1.plot(tt, Ep[si][random_ids].T*100, c='r', label='simulated (one example)')
        ax1.plot(tt, np.mean(Ep[si], axis=0)*100, c='b', ls='--', lw=2, label='mean')# and 95% CI
        #ax1.plot(tt, np.percentile(Ep[si],2.5,axis=0)*100, c='b')
        #ax1.plot(tt, np.percentile(Ep[si],97.5,axis=0)*100, c='b')
        ax1.plot(tt, P, c='k', label='actual')
        if model == 'lognormal':
            T = len(E[si])
            tt2 = np.arange(1,T+1)
            val = alpha[si] * np.exp(-(np.log(tt2)-mu[si])**2 / (2* sigma[si]**2))#+t0[i]
            val = sigmoid(val)
            ax1.plot(tt, val*100, c='m', label='no drug')
        ax1.legend()
        #ax1.set_xlabel('time (h)')
        ax1.set_ylabel('IIC burden (%)')
        ax1.set_ylim([-2,102])
        
        #TODO use imshow
        ax2 = fig.add_subplot(212)
        for di in range(ND):
            if np.max(Dscaled[si][:,di])>0:
                ax2.plot(tt, Dscaled[si][:,di]*Dmax[di], label=Dnames[di])
        ax2.legend()#ncol=2
        ax2.set_xlabel('time (h)')
        ax2.set_ylabel('Drug concentration')
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_dir, '%s.png'%sids[si]))
