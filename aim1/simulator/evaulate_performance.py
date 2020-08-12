from collections import defaultdict
import glob
import os
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.stats import pearsonr, binom
from tqdm import tqdm
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


with open('model_fit_sid2.pkl', 'rb') as f:
    model, fit = pickle.load(f)
with open('results.pickle', 'rb') as ff:
    res  = pickle.load(ff)
Ep_te_sim = res['Ep_te_sim']
Ep_te_baseline  = res['Ep_te_baseline']
sids = res['sids']

mat = sio.loadmat('CDE.mat')
C = mat['C']
D = mat['D']
E = mat['E']

perf_sim = defaultdict(list)
perf_bl = defaultdict(list)
Tcomb = 10
W = 900
for i, sid in enumerate(tqdm(sids)):
    Eobs = E[i][Tcomb:]
    Esim = Ep_te_sim[i][:,Tcomb:]
    Ebl = Ep_te_baseline[i][:,Tcomb:]
    
    notnan_ids = ~np.isnan(Eobs)
    Eobs = Eobs[notnan_ids]
    Esim = Esim[:,notnan_ids]
    Ebl = Ebl[:,notnan_ids]
    
    Esim_mean, Esim_lb, Esim_ub = np.percentile(Esim, (50,2.5,97.5), axis=0)
    Ebl_mean, Ebl_lb, Ebl_ub = np.percentile(Ebl, (50,2.5,97.5), axis=0)
    
    perf_sim['RMSE'].append(np.sqrt(np.mean((Esim_mean-Eobs)**2)))
    perf_bl['RMSE'].append(np.sqrt(np.mean((Ebl_mean-Eobs)**2)))
    perf_sim['CI95 Coverage'].append(np.mean( (Eobs>=Esim_lb)&(Eobs<=Esim_ub) ))
    perf_bl['CI95 Coverage'].append(np.mean( (Eobs>=Ebl_lb)&(Eobs<=Ebl_ub) ))
    perf_sim['LogLikelihood'].append(binom.logpmf(np.round(Eobs*W),W,np.clip(Esim_mean, 1e-6, 1-1e-6)).mean())
    perf_bl['LogLikelihood'].append(binom.logpmf(np.round(Eobs*W),W,np.clip(Ebl_mean, 1e-6, 1-1e-6)).mean())
        

metrics = ['RMSE', 'CI95 Coverage', 'LogLikelihood']

df_data = {'sid':sids}
for metric in metrics:
    print(metric)
    print(ttest_rel(perf_sim[metric], perf_bl[metric]))
    df_data['sim_'+metric] = perf_sim[metric]
    df_data['bl_'+metric] = perf_bl[metric]
df_perf = pd.DataFrame(data=df_data)
#df_perf = df_perf[['sid']+]
print(df_perf)
df_perf.to_csv('performance_comparison.csv', index=False)

for metric in metrics:
    """
    plt.close()
    plt.hist(rmse_sim, bins=np.linspace(0,1,11), color='b', label='Simulation', rwidth=0.8, alpha=0.2)
    plt.hist(rmse_bl, bins=np.linspace(0,1,11), color='k', label='Baseline', rwidth=0.8, alpha=0.2)
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel('RMSE')
    plt.tight_layout()
    plt.show()
    """

    plt.close()
    plt.scatter(perf_sim[metric], perf_bl[metric])
    min_ = min(min(perf_sim[metric]), min(perf_bl[metric]))
    max_ = max(max(perf_sim[metric]), max(perf_bl[metric]))
    plt.plot([min_, max_], [min_, max_], ls='--')
    plt.xlabel(metric+' from simulation')
    plt.ylabel(metric+' from Baseline')
    plt.tight_layout()
    plt.savefig(metric+'_scatterplot.png')
