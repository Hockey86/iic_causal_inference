import os
import pickle
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


W = 300
max_iter = 200
Dnames = ['lacosamide', 'levetiracetam', 'midazolam', 
          #'pentobarbital','phenobarbital',# 'phenytoin',
          'propofol', 'valproate']
ND = len(Dnames)
#models = ['lognormal']#, 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
models = ['normal_expit_ARMA16', 'student_t_expit_ARMA16', 'cauchy_expit_ARMA16', 'normal_probit_ARMA16']

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
    with open('../results/results_%s_iter%d.pickle'%(model, max_iter), 'rb') as ff:
        res  = pickle.load(ff)
    Psim = res['Psim']
    P = res['P']
    Dscaled = res['Dscaled']
    Dmax = res['Dmax']
    sids = res['sids']
    spec_db = res['spec']
    freq = res['freq']
    vmin, vmax = np.nanpercentile(np.concatenate([x.flatten() for x in spec_db]), (5, 95))
    
    for si, sid in enumerate(tqdm(sids)):
        T = len(P[si])
        tt = np.arange(T)*W*2/3600
        P_ = P[si]*100
        
        plt.close()
        fig = plt.figure(figsize=(12,8))
        
        ax1 = fig.add_subplot(311)
        ax1.imshow(spec_db[si].T, cmap='jet', aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax,
                   extent=(tt.min(), tt.max(), freq[si].min(), freq[si].max()))
        ax1.set_xlim([tt.min(), tt.max()])
        ax1.set_ylim([freq[si].min(), freq[si].max()])
        ax1.set_ylabel('freq (Hz)')
        
        ax2 = fig.add_subplot(312)
        random_ids = np.random.choice(len(Psim[si]), 1, replace=False)
        ax2.plot(tt, Psim[si][random_ids].T*100, c='r', label='simulated (one example)')
        ax2.plot(tt, np.mean(Psim[si], axis=0)*100, c='b', ls='--', lw=2, label='mean')# and 95% CI
        ax2.plot(tt, np.percentile(Psim[si],2.5,axis=0)*100, c='b', ls='--', label='95% CI')
        ax2.plot(tt, np.percentile(Psim[si],97.5,axis=0)*100, c='b', ls='--')
        ax2.plot(tt, P_, lw=2, c='k', alpha=0.5, label='actual')
        if model == 'lognormal':
            tt2 = np.arange(1,T+1)
            val = alpha[si] * np.exp(-(np.log(tt2)-mu[si])**2 / (2* sigma[si]**2))#+t0[i]
            val = sigmoid(val)
            ax2.plot(tt, val*100, c='m', label='no drug')
        ax2.legend(fontsize=12, frameon=False, ncol=3)
        #ax2.set_xlabel('time (h)')
        ax2.set_ylabel('IIC burden (%)')
        ax2.set_ylim([-2,102])
        ax2.set_xlim([tt.min(), tt.max()])
        
        #TODO use imshow
        ax3 = fig.add_subplot(313)
        for di in range(ND):
            if np.max(Dscaled[si][:,di])>0:
                ax3.plot(tt, Dscaled[si][:,di]*Dmax[di], label=Dnames[di])
        ax3.legend()#ncol=2
        ax3.set_xlabel('time (h)')
        ax3.set_ylabel('[drug]')
        ax3.set_xlim([tt.min(), tt.max()])
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_dir, '%s.png'%sids[si]))
        
