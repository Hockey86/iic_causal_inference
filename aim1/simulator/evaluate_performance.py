from itertools import product
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulator import *


models = ['normal_expit_ARMA16', 'cauchy_expit_ARMA16', 'student_t_expit_ARMA16', 'normal_probit_ARMA16', 'baseline']
max_iter = [200,200,200,200,100]
metrics = ['loglikelihood', 'CI95 Coverage']#, 'stRMSE'
W = 300
random_state = 2020

cluster = pd.read_csv('Cluster.csv', header=None)
cluster = np.argmax(cluster.values, axis=1)

perf = {}
for model, metric in product(models, metrics):
    with open('results/results_%s_iter%d.pickle'%(model, max_iter[models.index(model)]), 'rb') as ff:
        res = pickle.load(ff)
    Psim = res['Psim']
    P = res['P']
    Dscaled = res['Dscaled']
    sids = res['sids']
    
    if 'lognormal' in model:
        AR_T0 = 0
        MA_T0 = 6
        simulator = Simulator('stan_models/model_lognormal.stan', W, T0=[AR_T0, MA_T0], max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s_iter%d.pkl'%(model, max_iter[models.index(model)]))
    elif 'ARMA' in model:
        AR_T0 = int(model[-2:-1])
        MA_T0 = int(model[-1:])
        simulator = Simulator('stan_models/model_%s.stan'%model, W, T0=[AR_T0, MA_T0], max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s_iter%d.pkl'%(model, max_iter[models.index(model)]))
    elif model=='baseline':
        AR_T0 = 2
        simulator = BaselineSimulator(AR_T0, W, random_state=random_state)
    
    if metric in ['loglikelihood', 'CI95 Coverage']:
        perf[(model, metric)] = simulator.score(
                                    [x[AR_T0:] for x in Dscaled],
                                    [x[AR_T0:] for x in P],
                                    Psim=[x[:,AR_T0:] for x in Psim],
                                    method=metric)
        if perf[(model, metric)].ndim>1:
            perf_ = np.nanmean(perf[(model, metric)], axis=0)
            print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
        else:
            print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf[(model, metric)].mean(), np.nanpercentile(perf[(model, metric)], 2.5), np.nanpercentile(perf[(model, metric)], 97.5)))
        
    elif metric == 'WAIC':
        perf[(model, metric)] = simulator.score(
                                    [x[AR_T0:] for x in Dscaled],
                                    [x[AR_T0:] for x in P],
                                    Psim=[x[:,AR_T0:] for x in Psim],
                                    method=metric)
                                    
    elif metric == 'stRMSE':
        for TstRMSE in tqdm(range(2,13)):
            perf[(model, 'stRMSE(%d)'%TstRMSE)] = simulator.score(
                                        [x[AR_T0:] for x in Dscaled],
                                        [x[AR_T0:] for x in P],
                                        cluster=cluster,
                                        Ncluster=len(set(cluster)),
                                        method=metric, TstRMSE=TstRMSE)
        for TstRMSE in range(2,13):
            perf_ = np.nanmean(perf[(model, 'stRMSE(%d)'%TstRMSE)], axis=0)
            print('[%s, %s(%d)]: %.2f [%.2f -- %.2f]'%(model, metric, TstRMSE, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
            
import pdb;pdb.set_trace()
perf['sids'] = sids
with open('results/performance_metrics_%s.pickle'%str(models), 'wb') as ff:
    pickle.dump(perf, ff)
#plt.close();plt.plot(range(1,13),[np.nanmean(perf[('AR1','stRMSE_%d'%i)]) for i in range(1,13)], 'o-');plt.xlabel('step');plt.ylabel('AR1 stRMSE');plt.xticks(range(1,13));plt.grid(True);plt.show()

