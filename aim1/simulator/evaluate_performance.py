from itertools import product
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulator import *


models = ['baseline']#['lognormal', 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
metrics = ['stRMSE']#, 'loglikelihood', 'CI95 Coverage'
W = 300
K_MA = 6
max_iter = 50#1000
random_state = 2020

cluster = pd.read_csv('Cluster.csv', header=None)
cluster = np.argmax(cluster.values, axis=1)

perf = {}
for model, metric in product(models, metrics):
    with open('results/results_%s.pickle'%model, 'rb') as ff:
        res = pickle.load(ff)
    Ep = res['Ep_sim']
    E = res['E']
    P = res['P']
    Dscaled = res['Dscaled']
    sids = res['sids']
    
    if 'lognormal' in model:
        T0 = 0
        simulator = Simulator('stan_models/model_lognormal.stan', W, K_MA, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    elif 'AR' in model:
        T0 = int(model[-1:])
        simulator = Simulator('stan_models/model_%s.stan'%model, W, K_MA, T0=T0, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    elif model=='baseline':
        T0 = 2
        simulator = BaselineSimulator(T0, W, random_state=random_state)
    
    if metric=='loglikelihood':
        perf[(model, metric)] = simulator.score(
                                    [x[T0:] for x in Dscaled],
                                    [x[T0:] for x in E],
                                    Ep=[x[:,T0:] for x in Ep],
                                    method=metric)
        perf_ = np.nanmean(perf[(model, metric)], axis=0)
        print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
    elif metric=='stRMSE':
        for TstRMSE in tqdm(range(2,13)):
            perf[(model, 'stRMSE(%d)'%TstRMSE)] = simulator.score(
                                        [x[T0:] for x in Dscaled],
                                        [x[T0:] for x in E],
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

