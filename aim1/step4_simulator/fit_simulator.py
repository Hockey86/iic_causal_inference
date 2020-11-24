#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from simulator import *


if __name__=='__main__':

    #data_type = 'humanIIC'
    data_type = 'CNNIIC'
    
    with open(f'../data_to_fit_{data_type}.pickle', 'rb') as f:
        res = pickle.load(f)
    for k in res:
        exec(f'{k} = res["{k}"]')
    W = 300 #TODO include into pickle

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

    model_type = str(sys.argv[1])
    stan_path = f'stan_models/model_{model_type}.stan'
    
    max_iter = 100
    if model_type=='baseline':
        model_path = f'results/model_fit_{data_type}_{model_type}_iter{max_iter}.pkl'
        output_path = f'results/results_{data_type}_{model_type}_iter{max_iter}.pickle'
        AR_p = 2
        simulator = BaselineSimulator(AR_p, W, random_state=random_state)

    elif 'ARMA' in model_type:
        AR_p = int(sys.argv[2])
        MA_q = int(sys.argv[3])
        model_path = f'results/model_fit_{data_type}_{model_type}{AR_p},{MA_q}_iter{max_iter}.pkl'
        output_path = f'results/results_{data_type}_{model_type}{AR_p},{MA_q}_iter{max_iter}.pickle'
        simulator = Simulator(stan_path, W, T0=[AR_p, MA_q], max_iter=max_iter, random_state=random_state)
        
    simulator.fit(D, Pobs, cluster)
    simulator.save_model(model_path)
    #simulator.load_model(model_path)
    Psim = simulator.predict(D, cluster, Pstart=np.array([Pobs[i][:AR_p] for i in range(len(Pobs))]))

    import pdb;pdb.set_trace()
    with open(output_path, 'wb') as ff:
        pickle.dump({'Psim':Psim, 'Pobs':Pobs, 'Pname':Pname,
                     'Dscaled':D, 'Dmax':Dmax, 'Dname':Dname,
                     'C':C, 'Cname':Cname, 'W':W,
                     'cluster':cluster, 'sids':sids}, ff)
