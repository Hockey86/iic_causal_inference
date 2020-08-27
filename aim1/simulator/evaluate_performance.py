from itertools import product
import pickle
import numpy as np
from simulator import *


models = ['lognormal', 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
metrics = ['loglikelihood']#, 'stRMSE']#, 'CI95 Coverage'
W = 900
max_iter = 1000
random_state = 2020

perf = {}
for model, metric in product(models, metrics):
    with open('results/results_%s.pickle'%model, 'rb') as ff:
        res = pickle.load(ff)
    Ep = res['Ep_sim']
    E = res['E']
    Dscaled = res['Dscaled']
    sids = res['sids']
    
    if 'lognormal' in model:
        T0 = 0
        simulator = Simulator('stan_models/model_lognormal.stan', W, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    elif 'AR' in model:
        T0 = int(model[-1:])
        simulator = Simulator('stan_models/model_%s.stan'%model, W, T0=T0, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    
    if metric=='loglikelihood':
        perf[(model, metric)] = simulator.score(
                                    [x[T0:] for x in Dscaled],
                                    [x[T0:] for x in E],
                                    [x[:,T0:] for x in Ep],
                                    method=metric)
    elif metric=='stRMSE':
        for TstRMSE in range(1,13):
            perf[(model, 'stRMSE_%s'%TstRMSE)] = simulator.score(
                                        [x[T0:] for x in Dscaled],
                                        [x[T0:] for x in E],
                                        [x[:,T0:] for x in Ep],
                                        method=metric, TstRMSE=TstRMSE)
    perf_ = perf[(model, metric)].mean(axis=0)
    print('%s: %.2f [%.2f -- %.2f]'%((model, metric), perf_.mean(), np.percentile(perf_, 2.5), np.percentile(perf_, 97.5)))

perf['sids'] = sids
with open('performance_metrics.pickle', 'wb') as ff:
    pickle.dump(perf, ff)
