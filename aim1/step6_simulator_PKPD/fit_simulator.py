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

    D_halflife = {
        'lacosamide': [66, 30,96],     #  11h, (5-16h)
        'levetiracetam': [48, 36,60], #  8h, (6-10h)
        'midazolam': [15, 6,36],      #  2.5h, (1-6h)
        'pentobarbital': [195, 90,300], # 32.5h (15-50h)
        'phenobarbital': [474, 318,840], # 79h, (53-140h)
        'propofol': [2, 1,6],       # 20minutes, (10min-1h) (3-12h after long time) (needs 3 differential equations)
        'valproate': [96, 54,120],       # 16h, (9-20h)
        }
    D_halflife = np.array([D_halflife[x] for x in Dname])

    # standardize drugs
    Ddose_max = []
    for di in range(Ddose[0].shape[-1]):
        dd = np.concatenate([x[:,di] for x in Ddose])
        dd[dd==0] = np.nan
        Ddose_max.append(np.nanpercentile(dd,95))
    Ddose_max = np.array(Ddose_max)
    for i in range(len(Ddose)):
        Ddose[i] = Ddose[i]/Ddose_max

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
        
    simulator.fit_parallel(Ddose, D_halflife, Pobs, cluster, n_jobs=12)
    simulator.save_model(model_path)
    #simulator.load_model(model_path)
    Psim, Asim, Dsim = simulator.predict(Ddose, cluster, return_AD=True)
    #ii=202;plt.plot(Pobs[ii],c='k');plt.plot(Psim[ii].mean(axis=0),c='r');plt.plot(Ddose[ii],c='b');plt.show()
    
    import pdb;pdb.set_trace()
    with open(output_path, 'wb') as ff:
        pickle.dump({'Psim':Psim, 'Dsim':Dsim,
                     'Pobs':Pobs, 'Pname':Pname,
                     'Ddose':Ddose, 'Dname':Dname, 'Ddose_max':Ddose_max,
                     'C':C, 'Cname':Cname, 'W':W,
                    'cluster':cluster, 'sids':sids}, ff)
