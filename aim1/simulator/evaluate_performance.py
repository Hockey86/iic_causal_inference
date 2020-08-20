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


models = ['lognormal', 'AR2', 'AR1', 'baseline']
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
    
    if model in ['AR1', 'AR2']:
        T0 = int(model[2:])
        simulator = Simulator('stan_models/model_%s.stan'%model, W, T0=T0, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    elif model=='lognormal':
        T0 = 0
        simulator = Simulator('stan_models/model_lognormal.stan', W, max_iter=max_iter, random_state=random_state)
        simulator.load_model('results/model_fit_%s.pkl'%model)
    
    perf[(model, metric)] = simulator.score(
                                [x[T0:] for x in Dscaled],
                                [x[T0:] for x in E],
                                [x[:,T0:] for x in Ep],
                                method=metric)
    print((model, metric), perf[(model, metric)].mean())
    
