#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import sys
import numpy as np
from scipy.special import logit
from tqdm import tqdm
import matplotlib.pyplot as plt
from simulator import *


if __name__=='__main__':

    #data_type = 'humanIIC'
    data_type = 'CNNIIC'
    response_tostudy = sys.argv[1]  # 'iic_burden_smooth' or 'spike_rate'
    
    with open(f'../data_to_fit_{data_type}_iic_burden_smooth+spike_rate.pickle', 'rb') as f:
        res = pickle.load(f)
    for k in res:
        exec(f'{k} = res["{k}"]')
    Pobs = Pobs[response_tostudy]
    random_state = 2020

    # standardize drugs
    Dmax = []
    for di in range(D[0].shape[-1]):
        dd = np.concatenate([x[:,di] for x in D])
        dd[dd==0] = np.nan
        Dmax.append(np.nanpercentile(dd,95))
    Dmax = np.array(Dmax)
    for i in range(len(D)):
        D[i] = D[i]/Dmax
        
    # # define and infer model

    model_type = str(sys.argv[2])
    stan_path = f'stan_models/model_{model_type}.stan'
    
    max_iter = 1000
    if model_type=='baseline':
        model_path = f'results_{response_tostudy}/model_fit_{data_type}_{response_tostudy}_{model_type}_iter{max_iter}.pkl'
        output_path = f'results_{response_tostudy}/results_{data_type}_{response_tostudy}_{model_type}_iter{max_iter}.pickle'
        AR_p = 2
        simulator = BaselineSimulator(AR_p, W, random_state=random_state)

    elif 'ARMA' in model_type:
        AR_p = int(sys.argv[3])
        MA_q = int(sys.argv[4])
        model_path = f'results_{response_tostudy}/model_fit_{data_type}_{response_tostudy}_{model_type}{AR_p},{MA_q}_iter{max_iter}.pkl'
        output_path = f'results_{response_tostudy}/results_{data_type}_{response_tostudy}_{model_type}{AR_p},{MA_q}_iter{max_iter}.pickle'
        simulator = Simulator(stan_path, W, T0=[AR_p, MA_q], max_iter=max_iter, random_state=random_state)
        
    #simulator.fit(D, Pobs, cluster)
    simulator.fit_parallel(D, Pobs, cluster, n_jobs=12)
    simulator.save_model(model_path)
    #simulator.load_model(model_path)
    Pstart = np.array([Pobs[i][:AR_p] for i in range(len(Pobs))])
    Psim = simulator.predict(D, cluster, Astart=logit(np.clip(Pstart, 1e-6, 1-1e-6)))
    #ii=202;plt.plot(Pobs[ii],c='k');plt.plot(Psim[ii].mean(axis=0),c='r');plt.plot(D[ii],c='b');plt.show()
    
    import pdb;pdb.set_trace()
    with open(output_path, 'wb') as ff:
        pickle.dump({'Psim':Psim, 'Pobs':Pobs, 'Pname':Pname,
                     'Dscaled':D, 'Dmax':Dmax, 'Dname':Dname,
                     'C':C, 'Cname':Cname, 'W':W,
                    'cluster':cluster, 'sids':sids}, ff)
