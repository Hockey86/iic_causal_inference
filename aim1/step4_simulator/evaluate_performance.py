from itertools import product
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulator import *


#models = ['normal_expit_ARMA1,6', 'cauchy_expit_ARMA1,6', 'cauchy_expit_lognormal_ARMA1,6', 'cauchy_expit_lognormal_drugoutside_ARMA1,6', 'cauchy_expit_a0_as_lognormal_drugoutside_ARMA1,6', 'baseline']
models = ['cauchy_expit_lognormal_drugoutside_ARMA2,6', 'cauchy_expit_lognormal_drugoutside_ARMA3,6']
max_iter = [1000]*len(models)
metrics = ['WAIC', 'spearmanr', 'KL-divergence', 'loglikelihood', 'CI95 Coverage']#, 'stRMSE']
W = 300
random_state = 2020

perf = {}
for model, metric in product(models, metrics):
    with open('results/results_%s_iter%d.pickle'%(model, max_iter[models.index(model)]), 'rb') as ff:
        res = pickle.load(ff)
    Psim = res['Psim']
    Pobs = res['Pobs']
    Dscaled = res['Dscaled']
    sids = res['sids']
    
    if 'lognormal' in model:
        AR_T0 = 0
        MA_T0 = 6
        simulator = Simulator(None, W, T0=[AR_T0, MA_T0], max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s_iter%d.pkl'%(model, max_iter[models.index(model)]))
    elif 'ARMA' in model:
        AR_T0 = int(model[-2:-1])
        MA_T0 = int(model[-1:])
        simulator = Simulator(None, W, T0=[AR_T0, MA_T0], max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s_iter%d.pkl'%(model, max_iter[models.index(model)]))
    elif model=='baseline':
        AR_T0 = 2
        simulator = BaselineSimulator(AR_T0, W, random_state=random_state)
    
    if metric in ['loglikelihood', 'CI95 Coverage', 'KL-divergence', 'spearmanr']:
        perf[(model, metric)] = simulator.score(
                                    [x[AR_T0:] for x in Pobs],
                                    [x[:,AR_T0:] for x in Psim],
                                    metric)
        if perf[(model, metric)].ndim>1:
            perf_ = np.nanmean(perf[(model, metric)], axis=0)
            print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
        else:
            print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf[(model, metric)].mean(), np.nanpercentile(perf[(model, metric)], 2.5), np.nanpercentile(perf[(model, metric)], 97.5)))
        
    elif metric == 'WAIC':
        perf[(model, metric)] = simulator.waic
        print('[%s, %s]: %.2f'%(model, metric, perf[(model, metric)]))
                                    
    elif metric == 'stRMSE':
        cluster = pd.read_csv('Cluster.csv', header=None)
        cluster = np.argmax(cluster.values, axis=1)
        Ts = np.arange(1,25)
        perf[(model, metric)] = simulator.score(
                                Pobs, Psim, metric,
                                D=Dscaled,
                                cluster=cluster, Ncluster=len(set(cluster)),
                                Ts_stRMSE=Ts, Tinterval=5)
        perf[(model, metric)] = perf[(model, metric)].transpose(1,0,2)  #(#T, #pt, #posterior)
        perf['Ts_stRMSE'] = Ts
        for ii, t in enumerate(Ts):
            perf_ = np.nanmean(perf[(model, metric)][ii], axis=0)
            print('[%s, %s(%d)]: %.2f [%.2f -- %.2f]'%(model, metric, t, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
            
import pdb;pdb.set_trace()
perf['sids'] = sids
with open('results/performance_metrics_%s.pickle'%str(models), 'wb') as ff:
    pickle.dump(perf, ff)
#plt.close();plt.plot(range(1,13),[np.nanmean(perf[('AR1','stRMSE_%d'%i)]) for i in range(1,13)], 'o-');plt.xlabel('step');plt.ylabel('AR1 stRMSE');plt.xticks(range(1,13));plt.grid(True);plt.show()

