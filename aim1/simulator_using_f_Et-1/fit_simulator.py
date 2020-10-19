#!/usr/bin/env python
# coding: utf-8
from itertools import groupby
import os
import pickle
import sys
import timeit
import scipy.io as sio
from scipy.special import logit
from scipy.special import expit as sigmoid
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from myfunctions import preprocess, get_pk_k
from simulator import *


if __name__=='__main__':


    DATA_DIR = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
    #DATA_DIR = '/home/kentaro/Dropbox/step1_output/'

    sids = ['sid2', 'sid8', 'sid13', 'sid17', 'sid18', 'sid30', 'sid36', 'sid39', 'sid54',
            'sid56', 'sid69', 'sid77', 'sid82', 'sid88', 'sid91', 'sid92', 'sid297', 'sid327',
            'sid385', 'sid395', 'sid400', 'sid403', 'sid406', 'sid424', 'sid450', 'sid456',
            'sid490', 'sid512', 'sid551', 'sid557', 'sid734', 'sid736', 'sid801', 'sid821',
            'sid824', 'sid827', 'sid832', 'sid833', 'sid834', 'sid839', 'sid848', 'sid849',
            'sid852', 'sid872', 'sid876', 'sid880', 'sid881', 'sid884', 'sid886',
            'sid915', 'sid940', 'sid942', 'sid944', 'sid952', 'sid960', 'sid965', 'sid967',
            'sid983', 'sid987', 'sid988', 'sid994', 'sid1002', 'sid1006', 'sid1016', 'sid1022',
            'sid1025', 'sid1034', 'sid1038', 'sid1039', 'sid1055', 'sid1056', 'sid1063', 'sid1113',
            'sid1116', 'sid1337', 'sid1913', 'sid1915', 'sid1916', 'sid1917', 'sid1928', 'sid1956', 'sid1966']

    #sids = ['sid2', 'sid8', 'sid13', 'sid17', 'sid18', 'sid30']
    # exclude sid887 because there is no overlap between drug and IIC
    # , 'sid887'

    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam',
                    #'pentobarbital','phenobarbital',# 'phenytoin',
                    'propofol', 'valproate']


    # # define PK time constants for each drug
    PK_K = get_pk_k()

    W = 300
    Pobs = []
    D = []
    C = []
    spec = []
    freq = []
    for sid in tqdm(sids):
        Pobs_, Pname, D_, Dname, C_, Cname, spec_, freq_ = preprocess(sid, DATA_DIR, PK_K, W, drugs_tostudy)
        Pobs.append(Pobs_)
        D.append(D_)
        C.append(C_)
        spec.append(spec_)
        freq.append(freq_)

    C = np.array(C)
    #sids = C[:,0].astype(str)
    C = C[:,1:].astype(float)
    Cname = Cname[1:]

    # get cluster
    #TODO K-means
    cluster = pd.read_csv('Cluster.csv', header=None)
    cluster = np.argmax(cluster.values, axis=1)#[:len(sids)]

    # exclude patients with flat IIC
    std_thres = 0.01
    keep_ids = [i for i in range(len(sids)) if np.nanstd(Pobs[i])>std_thres]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    C = C[keep_ids]
    spec = [spec[i] for i in keep_ids]
    freq = [freq[i] for i in keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    print('%d patients'%len(sids))


    # # print stats of the data

    # In[8]:


    #plt.plot(E[0])
    print(sorted([len(x) for x in D]))

    """
    for i in tqdm(range(len(sids))):
        plt.close()
        plt.plot(Pobs[i])
        plt.title(sids[i])
        plt.savefig('E_figures/%s.png'%sids[i])
    """


    # # remove long gaps in data
    # In[8]:
    if W==900:
        ind = sids.index('sid1038') # T=1506
        Pobs[ind] = Pobs[ind][1469:]
        D[ind] = D[ind][1469:]
        spec[ind] = spec[ind][1469:]

        ind = sids.index('sid91') # T=657
        Pobs[ind] = Pobs[ind][602:]
        D[ind] = D[ind][602:]
        spec[ind] = spec[ind][602:]

        ind = sids.index('sid30') # T=453
        Pobs[ind] = Pobs[ind][395:]
        D[ind] = D[ind][395:]
        spec[ind] = spec[ind][395:]

        ind = sids.index('sid1966') # T=334
        Pobs[ind] = Pobs[ind][300:]
        D[ind] = D[ind][300:]
        spec[ind] = spec[ind][300:]

        ind = sids.index('sid395') # T=247
        Pobs[ind] = Pobs[ind][196:]
        D[ind] = D[ind][196:]
        spec[ind] = spec[ind][196:]

        ind = sids.index('sid1025') # T=245
        Pobs[ind] = Pobs[ind][178:]
        D[ind] = D[ind][178:]
        spec[ind] = spec[ind][178:]

        ind = sids.index('sid36')
        Pobs[ind] = Pobs[ind][:48]
        D[ind] = D[ind][:48]
        spec[ind] = spec[ind][:48]

        ind = sids.index('sid801')
        Pobs[ind] = Pobs[ind][33:]
        D[ind] = D[ind][33:]
        spec[ind] = spec[ind][33:]

        ind = sids.index('sid960')
        Pobs[ind] = Pobs[ind][72:]
        D[ind] = D[ind][72:]
        spec[ind] = spec[ind][72:]

        ind = sids.index('sid1006')
        Pobs[ind] = Pobs[ind][47:]
        D[ind] = D[ind][47:]
        spec[ind] = spec[ind][47:]

        ind = sids.index('sid1022')
        Pobs[ind] = Pobs[ind][:39]
        D[ind] = D[ind][:39]
        spec[ind] = spec[ind][:39]

        ind = sids.index('sid456')
        Pobs[ind] = Pobs[ind][99:137]
        D[ind] = D[ind][99:137]
        spec[ind] = spec[ind][99:137]

        ind = sids.index('sid965')
        Pobs[ind] = Pobs[ind][88:]
        D[ind] = D[ind][88:]
        spec[ind] = spec[ind][88:]

        ind = sids.index('sid915')
        Pobs[ind] = Pobs[ind][93:]
        D[ind] = D[ind][93:]
        spec[ind] = spec[ind][93:]

    elif W==300:
        if 'sid1038' in sids:
            ind = sids.index('sid1038') # T=1506
            Pobs[ind] = Pobs[ind][4407:]
            D[ind] = D[ind][4407:]
            spec[ind] = spec[ind][4407:]

        if 'sid91' in sids:
            ind = sids.index('sid91') # T=657
            Pobs[ind] = Pobs[ind][1811:]
            D[ind] = D[ind][1811:]
            spec[ind] = spec[ind][1811:]

        if 'sid30' in sids:
            ind = sids.index('sid30') # T=453
            Pobs[ind] = Pobs[ind][1189:]
            D[ind] = D[ind][1189:]
            spec[ind] = spec[ind][1189:]

        if 'sid1966' in sids:
            ind = sids.index('sid1966') # T=334
            Pobs[ind] = Pobs[ind][902:]
            D[ind] = D[ind][902:]
            spec[ind] = spec[ind][902:]

        if 'sid395' in sids:
            ind = sids.index('sid395') # T=247
            Pobs[ind] = Pobs[ind][588:]
            D[ind] = D[ind][588:]
            spec[ind] = spec[ind][588:]

        if 'sid1025' in sids:
            ind = sids.index('sid1025') # T=245
            Pobs[ind] = Pobs[ind][536:]
            D[ind] = D[ind][536:]
            spec[ind] = spec[ind][536:]

        if 'sid36' in sids:
            ind = sids.index('sid36')
            Pobs[ind] = Pobs[ind][:144]
            D[ind] = D[ind][:144]
            spec[ind] = spec[ind][:144]

        if 'sid801' in sids:
            ind = sids.index('sid801')
            Pobs[ind] = Pobs[ind][103:]
            D[ind] = D[ind][103:]
            spec[ind] = spec[ind][103:]

        if 'sid960' in sids:
            ind = sids.index('sid960')
            Pobs[ind] = Pobs[ind][217:]
            D[ind] = D[ind][217:]
            spec[ind] = spec[ind][217:]

        if 'sid1006' in sids:
            ind = sids.index('sid1006')
            Pobs[ind] = Pobs[ind][144:]
            D[ind] = D[ind][144:]
            spec[ind] = spec[ind][144:]

        if 'sid1022' in sids:
            ind = sids.index('sid1022')
            Pobs[ind] = Pobs[ind][:116]
            D[ind] = D[ind][:116]
            spec[ind] = spec[ind][:116]

        if 'sid456' in sids:
            ind = sids.index('sid456')
            Pobs[ind] = Pobs[ind][298:411]
            D[ind] = D[ind][298:411]
            spec[ind] = spec[ind][298:411]

        if 'sid965' in sids:
            ind = sids.index('sid965')
            Pobs[ind] = Pobs[ind][264:]
            D[ind] = D[ind][264:]
            spec[ind] = spec[ind][264:]

        if 'sid915' in sids:
            ind = sids.index('sid915')
            Pobs[ind] = Pobs[ind][279:]
            D[ind] = D[ind][279:]
            spec[ind] = spec[ind][279:]
    else:
        raise ValueError('W=%d'%W)
    print(sorted([len(x) for x in D]))

    # # remove flat drug at the beginning or end



    # In[9]:


    for i in range(len(sids)):
        d = D[i].sum(axis=1)

        start = 0
        for gi, g in enumerate(groupby(d)):
            if gi==0:
                j, k = g
                ll = len(list(k))
                if j==0:
                    start = ll
            else:
                break

        end = 0
        for gi, g in enumerate(groupby(d[::-1])):
            if gi==0:
                j, k = g
                ll = len(list(k))
                if j==0:
                    end = ll
            else:
                break
        end = len(d)-end

        Pobs[i] = Pobs[i][start:end]
        D[i] = D[i][start:end]
        spec[i] = spec[i][start:end]

    print(sorted([len(x) for x in D]))


    # # make sure the first T0 points are not NaN to initialize the model

    # In[10]:


    T0 = 2
    for i in range(len(sids)):
        # move along time to find the first time when 2 points are not NaN
        found = False
        for t in range(len(D[i])-T0):
            if all([not np.isnan(Pobs[i][t+j]) for j in range(T0)]):
                found = True
                break
        if not found:
            print(sids[i], 'not found first T0 points being not NaN.')
            continue
        Pobs[i] = Pobs[i][t:]
        D[i] = D[i][t:]
        spec[i] = spec[i][t:]

    print(sorted([len(x) for x in D]))

    random_state = 2020

    # standardize features
    """
    Cmean = C.mean(axis=0)
    Cstd = C.std(axis=0)
    C = (C-Cmean)/Cstd
    """
    Dmax = []
    for di in range(D[0].shape[-1]):
        dd = np.concatenate([x[:,di] for x in D])
        dd[dd==0] = np.nan
        Dmax.append(np.nanpercentile(dd,95))
    Dmax = np.array(Dmax)
    for i in range(len(D)):
        D[i] = D[i]/Dmax

    with open('data_to_fit.pickle', 'wb') as f:
        pickle.dump({'D':D, 'Dname':Dname, 'Dmax':Dmax,
                     'Pobs':Pobs, 'Pname':Pname,
                     'cluster':cluster, 'sids':sids,
                     'C':C, 'Cname':Cname}, f)

    # # define and infer model

    model_type = str(sys.argv[1])

    max_iter = 1000
    stan_path = 'stan_models/model_%s.stan'%model_type
    model_path = 'results/model_fit_%s_iter%d.pkl'%(model_type, max_iter)
    if model_type=='baseline':
        simulator = BaselineSimulator(2, W, random_state=random_state)
        simulator.fit(D, Pobs)
        Psim = simulator.predict(D, Pobs)

    elif 'lognormal' in model_type:
        MA_q = 6
        simulator = Simulator(stan_path, W, T0=[0, MA_q], max_iter=max_iter, random_state=random_state)
        simulator.fit(D, Pobs, cluster)
        simulator.save_model(model_path)
        #simulator.load_model(model_path)
        Psim = simulator.predict(D, cluster)

    elif 'ARMA' in model_type:
        AR_p = int(model_type[-2:-1])
        MA_q = int(model_type[-1:])
        simulator = Simulator(stan_path, W, T0=[AR_p, MA_q], max_iter=max_iter, random_state=random_state)
        simulator.fit(D, Pobs, cluster, loss_weight=1)
        simulator.save_model(model_path)
        #simulator.load_model(model_path)
        Psim = simulator.predict(D, cluster, Pstart=np.array([Pobs[i][:AR_p] for i in range(len(Pobs))]))

    with open('results/results_%s_iter%d.pickle'%(model_type, max_iter), 'wb') as ff:
        pickle.dump({'Psim':Psim, 'Pobs':Pobs, 'Pname':Pname,
                     'Dscaled':D, 'Dmax':Dmax, 'Dname':Dname,
                     'C':C, 'Cname':Cname,
                     'spec':spec, 'freq':freq,
                     'cluster':cluster, 'sids':sids}, ff)
