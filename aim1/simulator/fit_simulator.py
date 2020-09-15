#!/usr/bin/env python
# coding: utf-8

# # import stuff

# In[1]:


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
sns.set()
from simulator import *


# In[2]:


DATA_DIR = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'


# # define PK time constants for each drug

# In[3]:


halflife = pd.DataFrame({
    'lacosamide':[13],
    'levetiracetam':[6],
    'midazolam':[1.5],
    'pentobarbital':[15],
    'phenobarbital':[53],
    'phenytoin':[22],
    'propofol':[1.5],
    'valproate':[8]
    },index=['t1/2'])

halflife = halflife.append(np.log(2) / halflife.rename(index={'t1/2':'k'}))

W = 300

# In[4]:


def logsigmoid(x):
    """
    Computes the log(sigmoid(x))
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec2
    """
    x = np.array(x)
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    if type(x)==float or x.ndim==0:
        out = float(out)
    return out
    

def drug_concentration(d_ts,k):
    """
    d_ts.shape = (#drug, T)
    """
    k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
    conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
    conc = conc[:,:d_ts.shape[1]]
    return conc


# # data functions

# In[5]:


def patient(file, W):
    window = W
    step   = W
    
    #if '.mat' in file:
    s = sio.loadmat(os.path.join(DATA_DIR, file))
    human_iic = s['human_iic'][0].astype(float)
    spike = s['spike'][0].astype(float)
    drugs = s['drugs_weightnormalized'].astype(float)
    artifact = s['artifact'][0].astype(float)
    human_iic[artifact==1] = np.nan
    spike[artifact==1] = np.nan

    drugnames = list(map(lambda x: x.strip(), s['Dnames']))
    drugs_window = np.array([ np.mean(drugs[i:i+window],axis=0) for i in range(0,len(drugs),step) ])

    sz_burden = (human_iic==1).astype(float)
    sz_burden[np.isnan(human_iic)] = np.nan
    sz_burden_window = [np.nanmean(sz_burden[i:i+window]) for i in range(0, len(sz_burden),step)]

    iic_burden = np.in1d(human_iic, [1,2,3,4]).astype(float)
    iic_burden[np.isnan(human_iic)] = np.nan
    iic_burden_window = [np.nanmean(iic_burden[i:i+window]) for i in range(0, len(iic_burden),step)]

    spike_rate_window = [np.nanmean(spike[i:i+window]) for i in range(0, len(spike),step)]

    df = pd.DataFrame(data=np.c_[sz_burden_window, iic_burden_window, spike_rate_window, drugs_window],
                      columns=['sz_burden', 'iic_burden', 'spike_rate']+drugnames)
    return df


# In[6]:


def preprocess(sid):  # previsously called patient_data
    PK_K = halflife.loc['k'].to_numpy()

    #fetch the data
    file = sid + '.mat'
    p = patient(file, W)

    #setting up the data
    response_tostudy = ['iic_burden']
    Eobs = p[response_tostudy].values.flatten()

    #PK
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                    #'pentobarbital','phenobarbital',# 'phenytoin',
                    'propofol', 'valproate']
    Ddose = p[drugs_tostudy].fillna(0).to_numpy().T
    D = drug_concentration(Ddose, PK_K).T

    #cov_tostudy = ['Age']
    C = pd.read_csv(os.path.join(DATA_DIR, 'covariates.csv'))
    Cname = list(C.columns)
    C = C[C.Index==sid].iloc[0]#[cov_tostudy]
    
    #Eobs.shape = (T,)
    #D.shape = (T,#drug)
    #C.shape = (#covaraites,)
    return Eobs, D, C, Cname


# # generate data

# In[7]:


Pobs = []
Eobs = []
D = []
C = []
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

# exclude sid887 because there is no overlap between drug and IIC
# , 'sid887'

for sid in tqdm(sids):
    Pobs_, D_, C_, Cname = preprocess(sid)
    Eobs_ = Pobs_*W
    Eobs_[np.isnan(Eobs_)] = -1   # convert NaN to -1 for int dtype
    Eobs.append(np.round(Eobs_).astype(int))
    Pobs.append(np.clip(Pobs_, 1e-6, 1-1e-6))
    D.append(D_)
    C.append(C_)

C = np.array(C)
#sids = C[:,0].astype(str)
C = C[:,1:].astype(float)
Cname = Cname[1:]
#sio.savemat('C.mat', {'C':C, 'Cname':Cname})

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
    Eobs[ind] = Eobs[ind][1469:]
    D[ind] = D[ind][1469:]

    ind = sids.index('sid91') # T=657
    Pobs[ind] = Pobs[ind][602:]
    Eobs[ind] = Eobs[ind][602:]
    D[ind] = D[ind][602:]

    ind = sids.index('sid30') # T=453
    Pobs[ind] = Pobs[ind][395:]
    Eobs[ind] = Eobs[ind][395:]
    D[ind] = D[ind][395:]

    ind = sids.index('sid1966') # T=334
    Pobs[ind] = Pobs[ind][300:]
    Eobs[ind] = Eobs[ind][300:]
    D[ind] = D[ind][300:]

    ind = sids.index('sid395') # T=247
    Pobs[ind] = Pobs[ind][196:]
    Eobs[ind] = Eobs[ind][196:]
    D[ind] = D[ind][196:]

    ind = sids.index('sid1025') # T=245
    Pobs[ind] = Pobs[ind][178:]
    Eobs[ind] = Eobs[ind][178:]
    D[ind] = D[ind][178:]

    ind = sids.index('sid36')
    Pobs[ind] = Pobs[ind][:48]
    Eobs[ind] = Eobs[ind][:48]
    D[ind] = D[ind][:48]

    ind = sids.index('sid801')
    Pobs[ind] = Pobs[ind][33:]
    Eobs[ind] = Eobs[ind][33:]
    D[ind] = D[ind][33:]

    ind = sids.index('sid960')
    Pobs[ind] = Pobs[ind][72:]
    Eobs[ind] = Eobs[ind][72:]
    D[ind] = D[ind][72:]

    ind = sids.index('sid1006')
    Pobs[ind] = Pobs[ind][47:]
    Eobs[ind] = Eobs[ind][47:]
    D[ind] = D[ind][47:]

    ind = sids.index('sid1022')
    Pobs[ind] = Pobs[ind][:39]
    Eobs[ind] = Eobs[ind][:39]
    D[ind] = D[ind][:39]

    ind = sids.index('sid456')
    Pobs[ind] = Pobs[ind][99:137]
    Eobs[ind] = Eobs[ind][99:137]
    D[ind] = D[ind][99:137]

    ind = sids.index('sid965')
    Pobs[ind] = Pobs[ind][88:]
    Eobs[ind] = Eobs[ind][88:]
    D[ind] = D[ind][88:]

    ind = sids.index('sid915')
    Pobs[ind] = Pobs[ind][93:]
    Eobs[ind] = Eobs[ind][93:]
    D[ind] = D[ind][93:]
elif W==300:
    ind = sids.index('sid1038') # T=1506
    Pobs[ind] = Pobs[ind][4407:]
    Eobs[ind] = Eobs[ind][4407:]
    D[ind] = D[ind][4407:]

    ind = sids.index('sid91') # T=657
    Pobs[ind] = Pobs[ind][1811:]
    Eobs[ind] = Eobs[ind][1811:]
    D[ind] = D[ind][1811:]

    ind = sids.index('sid30') # T=453
    Pobs[ind] = Pobs[ind][1189:]
    Eobs[ind] = Eobs[ind][1189:]
    D[ind] = D[ind][1189:]

    ind = sids.index('sid1966') # T=334
    Pobs[ind] = Pobs[ind][902:]
    Eobs[ind] = Eobs[ind][902:]
    D[ind] = D[ind][902:]

    ind = sids.index('sid395') # T=247
    Pobs[ind] = Pobs[ind][588:]
    Eobs[ind] = Eobs[ind][588:]
    D[ind] = D[ind][588:]

    ind = sids.index('sid1025') # T=245
    Pobs[ind] = Pobs[ind][536:]
    Eobs[ind] = Eobs[ind][536:]
    D[ind] = D[ind][536:]

    ind = sids.index('sid36')
    Pobs[ind] = Pobs[ind][:144]
    Eobs[ind] = Eobs[ind][:144]
    D[ind] = D[ind][:144]

    ind = sids.index('sid801')
    Pobs[ind] = Pobs[ind][103:]
    Eobs[ind] = Eobs[ind][103:]
    D[ind] = D[ind][103:]

    ind = sids.index('sid960')
    Pobs[ind] = Pobs[ind][217:]
    Eobs[ind] = Eobs[ind][217:]
    D[ind] = D[ind][217:]

    ind = sids.index('sid1006')
    Pobs[ind] = Pobs[ind][144:]
    Eobs[ind] = Eobs[ind][144:]
    D[ind] = D[ind][144:]

    ind = sids.index('sid1022')
    Pobs[ind] = Pobs[ind][:116]
    Eobs[ind] = Eobs[ind][:116]
    D[ind] = D[ind][:116]

    ind = sids.index('sid456')
    Pobs[ind] = Pobs[ind][298:411]
    Eobs[ind] = Eobs[ind][298:411]
    D[ind] = D[ind][298:411]

    ind = sids.index('sid965')
    Pobs[ind] = Pobs[ind][264:]
    Eobs[ind] = Eobs[ind][264:]
    D[ind] = D[ind][264:]

    ind = sids.index('sid915')
    Pobs[ind] = Pobs[ind][279:]
    Eobs[ind] = Eobs[ind][279:]
    D[ind] = D[ind][279:]
else:
    raise ValueError('W=%d'%W)
"""
remove_sids = ['sid1038', 'sid91', 'sid30', 'sid1966', 'sid395',
               'sid1025', 'sid36', 'sid801', 'sid960', 'sid1006',
               'sid1022', 'sid456', 'sid965', 'sid915']
keep_ids = [i for i in range(len(sids)) if sids[i] not in remove_sids]
Pobs = [Pobs[i] for i in keep_ids]
Eobs = [Eobs[i] for i in keep_ids]
D = [D[i] for i in keep_ids]
C = C[keep_ids]
sids = [sids[i] for i in keep_ids]
"""
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
    Eobs[i] = Eobs[i][start:end]
    D[i] = D[i][start:end]
    
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
    Eobs[i] = Eobs[i][t:]
    Pobs[i] = Pobs[i][t:]
    D[i] = D[i][t:]
    
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

# # define and infer model

# get cluster
#TODO K-means
cluster = pd.read_csv('Cluster.csv', header=None)
cluster = np.argmax(cluster.values, axis=1)#[:len(sids)]
#cluster = cluster[keep_ids]
#cluster = LabelEncoder().fit_transform(cluster)

model_type = str(sys.argv[1])

max_iter = 50#0
save_path = 'results/model_fit_%s.pkl'%model_type
if model_type=='baseline':
    simulator = BaselineSimulator(2, W, random_state=random_state)
    simulator.fit(D, Eobs)
    Ep_sim = simulator.predict(D, Pobs)
    
elif 'lognormal' in model_type:
    MA_T0 = 6
    simulator = Simulator('stan_models/model_%s.stan'%model_type, W, T0=[0, MA_T0], max_iter=max_iter, random_state=random_state)
    #simulator.load_model(save_path)
    simulator.fit(D, Eobs, cluster, save_path=save_path)
    Ep_sim = simulator.predict(D, cluster)
    
elif 'ARMA' in model_type:
    AR_T0 = int(model_type[-2:-1])
    MA_T0 = int(model_type[-1:])
    simulator = Simulator('stan_models/model_%s.stan'%model_type, W, T0=[AR_T0, MA_T0], max_iter=max_iter, random_state=random_state)
    #simulator.load_model(save_path)
    simulator.fit(D, Eobs, cluster, save_path=save_path)
    Ep_sim = simulator.predict(D, cluster, Pstart=np.array([Pobs[i][:AR_T0] for i in range(len(Pobs))]))

with open('results/results_%s.pickle'%model_type, 'wb') as ff:
    pickle.dump({'Ep_sim':Ep_sim,
                 'E':Eobs, 'P':Pobs,
                 'Dscaled':D, 'Dmax':Dmax,
                 'sids':sids}, ff)

