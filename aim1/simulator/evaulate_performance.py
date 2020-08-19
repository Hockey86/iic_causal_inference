from collections import defaultdict
from itertools import product
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
from simulator import *


models = ['lognormal', 'baseline']#, 'AR2'
metrics = ['loglikelihood']#, 'stRMSE']#, 'CI95 Coverage'
W = 900
max_iter = 1000
random_state = 2020

res = sio.loadmat('data_before_fit.mat')
C = res['C']
D = res['D']
E = res['E']
P = res['P']
D = [D[0,i] for i in range(D.shape[1])]
E = [E[0,i] for i in range(E.shape[1])]
P = [P[0,i] for i in range(P.shape[1])]

perf = {}
for model, metric in product(models, metrics):
    with open('results/results_%s.pickle'%model, 'rb') as ff:
        res = pickle.load(ff)
        Ep = res['Ep_sim']
    
    if model in ['AR1', 'AR2']:
        T0 = int(model_type[2:])
        simulator = Simulator('stan_models/model_%s.stan'%model_type, W, T0=T0, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    elif model=='lognormal':
        simulator = Simulator('stan_models/model_lognormal.stan', W, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    
    perf[(model, metric)] = simulator.score(D, E, Ep, method=metric)#, return_list=True)

import pdb;pdb.set_trace()
a=1
