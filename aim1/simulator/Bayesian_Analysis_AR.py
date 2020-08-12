#!/usr/bin/env python
# coding: utf-8

# # import stuff

# In[1]:


from itertools import groupby
import os
import timeit
import scipy.io as sio
from scipy.special import logit
from scipy.special import expit as sigmoid
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import pystan


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


def patient(file):
    window = 900
    step   = 900
    
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
    p = patient(file)

    #setting up the data
    response_tostudy = ['iic_burden']
    Eobs = p[response_tostudy].values.flatten()

    #PK
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                    'pentobarbital','phenobarbital',# 'phenytoin',
                    'propofol', 'valproate']
    Ddose = p[drugs_tostudy].fillna(0).to_numpy().T
    D = drug_concentration(Ddose, PK_K).T

    cov_tostudy = ['Age']
    C = pd.read_csv(os.path.join(DATA_DIR, 'covariates.csv'))
    C = C[C.Index==sid][cov_tostudy].iloc[0]
    
    #Eobs.shape = (T,)
    #D.shape = (T,#drug)
    #C.shape = (#covaraites,)
    return Eobs, D, C


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

W=900
for sid in tqdm(sids):
    Pobs_, D_, C_ = preprocess(sid)
    Eobs_ = Pobs_*W
    Eobs_[np.isnan(Eobs_)] = -1   # convert NaN to -1 for int dtype
    Eobs.append(np.round(Eobs_).astype(int))
    Pobs.append(np.clip(Pobs_, 1e-6, 1-1e-6))
    D.append(D_)
    C.append(C_)

C = np.array(C)


# # print stats of the data

# In[8]:


#plt.plot(E[0])
print(Pobs[0])
print(Eobs[0])
print(C.shape)
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


# # pad to same length

# In[11]:


maxT = np.max([len(x) for x in D])

for i in range(len(sids)):
    Eobs[i] = np.r_[Eobs[i], np.zeros(maxT-len(Eobs[i]), dtype=int)-1]
    Pobs[i] = np.r_[Pobs[i], np.zeros(maxT-len(Pobs[i]))+np.nan]
    D[i] = np.r_[D[i], np.zeros((maxT-len(D[i]), D[i].shape[1]))]

Eobs = np.array(Eobs)
Pobs = np.array(Pobs)
D = np.array(D)
N = len(D)
T = D.shape[1]
Ts = np.array([np.sum(~np.isnan(x)) for x in Pobs])

print(Eobs.shape)
print(Pobs.shape)
print(D.shape)
print(C.shape)

#sio.savemat('CDE.mat', {'C':C, 'D':D, 'E':Pobs})

# # define model

# In[12]:


import pickle
from hashlib import md5

def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        model = pickle.load(open(cache_fn, 'rb'))
    except:
        model = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(model, f)
    else:
        print("Using cached StanModel")
    return model


# In[15]:

with open('stan_models/model_AR.stan', 'r') as f:
    pd_model = f.read()
model = StanModel_cache(model_code=pd_model)


# # infer the model

# In[ ]:

random_state = 2020

# leave one patient out cross-validation
fit_res = []
Ep_tr_sim = []
Ep_te_sim = []
Ep_te_baseline = []
E_te = []
D_te = []
sids_te = []
for si, sid in enumerate(sids):
    print('\n############### [%d/%d] %s ###############\n'%(si+1, len(sids), sid))
    sids_te.append(sid)
    
    trids = [x for x in range(len(sids)) if sids[x]!=sid]
    Eobstr = Eobs[trids]
    Pobstr = Pobs[trids]
    Dtr = D[trids]
    Ctr = C[trids]
    Ttr = Ts[trids]
    
    teid = [si]
    Eobste = np.array(Eobs[teid])
    Pobste = np.array(Pobs[teid])
    Dte = np.array(D[teid])
    Cte = np.array(C[teid])
    #Tte
    
    # standardize features
    Cmean = Ctr.mean(axis=0)
    Cstd = Ctr.std(axis=0)
    Ctr = (Ctr-Cmean)/Cstd
    Cte = (Cte-Cmean)/Cstd
    #Ctr = Ctr/100
    #Cte = Cte/100
    Dtr_nan = np.array(Dtr)
    Dtr_nan[Dtr_nan==0] = np.nan
    Dmax  = np.nanpercentile(Dtr_nan, 95, axis=(0,1))
    Dtr = Dtr/Dmax
    Dte = Dte/Dmax
    
    # combine the first Tcomb steps of testing to training
    # to enable patient-specific training
    Eobstr2 = np.r_[Eobstr, Eobste]
    Pobstr2 = np.r_[Pobstr, Pobste]
    Dtr2 = np.r_[Dtr, Dte]
    Ctr2 = np.r_[Ctr, Cte]
    Tcomb = 10  # must be >T0
    Eobste[:,Tcomb:] = -1
    Pobste[:,Tcomb:] = np.nan
    Dte[:,Tcomb:] = 0
    Ttr = np.r_[Ttr, [np.sum(~np.isnan(Pobste[ii])) for ii in range(len(Pobste))]]
    Eobstr = np.r_[Eobstr, Eobste]
    Pobstr = np.r_[Pobstr, Pobste]
    Dtr = np.r_[Dtr, Dte]
    Ctr = np.r_[Ctr, Cte]
    
    Eobstr_flatten = Eobstr[:,T0:].flatten()#
    not_empty_ids = np.where(Eobstr_flatten!=-1)[0]
    not_empty_num = len(not_empty_ids)
    Eobstr_flatten_nonan = Eobstr_flatten[not_empty_ids]
    
    # generate sample weights that balances different lengths
    sample_weights = np.zeros_like(Eobstr[:,T0:]) + 1/(Ttr-T0).reshape(-1,1)#
    sample_weights = sample_weights.flatten()[not_empty_ids]
    sample_weights = sample_weights/sample_weights.mean()
    
    data_feed = {'W':W,
                 'N':N,
                 'T':T,
                 'T0':T0,
                 'ND':Dtr.shape[-1],
                 #'NC':Ctr.shape[-1],
                 'not_empty_num':not_empty_num,
                 'not_empty_ids':not_empty_ids+1,  # +1 for stan
                 'sample_weights':sample_weights,
                 'Eobs_flatten_nonan':Eobstr_flatten_nonan,
                 'D':Dtr.transpose(1,0,2),  # because matrix[N,ND] D[T];
                 #'C':Ctr,
                 #'p_start':Pobstr[:,:T0],
                 'A_start':logit(Pobstr[:,:T0]),
                 }
    model = StanModel_cache(model_code=pd_model)
    
    # sampling
    
    # try multiple times with few iterations
    # the fastest one is usually the one that converges
    times = []
    for rs in range(3):
        print('Try %d'%(rs+1,))
        st = timeit.default_timer()
        fit = model.sampling(data=data_feed, iter=100, verbose=True, chains=1, seed=random_state+rs)
        et = timeit.default_timer()
        times.append(et-st)
        
    # sample many iterations for the one that can converge
    fit = model.sampling(data=data_feed, iter=1000, verbose=True, chains=1, seed=random_state+np.argmin(times))
                      #control={'max_treedepth':9})#, 'adapt_delta':0.9})
    print(fit.stansummary(pars=['mu_a0','mu_a1','mu_a2','mu_b',
                                'sigma_a0','sigma_a1','sigma_a2','sigma_b']))
    
    df = fit.to_dataframe(pars=['a0','a1','a2','b','sigma_a0','sigma_a1','sigma_a2','sigma_b'])
    
    # save
    df.to_csv('fit_dataframe_%s.csv'%sid, index=False)
    with open('model_fit_%s.pkl'%sid, 'wb') as f:
        pickle.dump([model, fit], f)
    """
    with open('model_fit_%s.pkl'%sid, 'rb') as f:
        model, fit = pickle.load(f)
    df = pd.read_csv('fit_dataframe_%s.csv'%sid)
    """
    fit_res.append(fit)
    
    # predict
    Epte = []
    Ppte = []
    start = 0#len(df)//2
    for i in tqdm(range(start,len(df))):
        a0 = np.array([df['a0[%d]'%ii].iloc[i] for ii in range(1,len(Dtr)+1)])
        a1 = np.array([df['a1[%d]'%ii].iloc[i] for ii in range(1,len(Dtr)+1)])
        a2 = np.array([df['a2[%d]'%ii].iloc[i] for ii in range(1,len(Dtr)+1)])
        b = np.array([[df['b[%d,%d]'%(jj,ii)].iloc[i] for ii in range(1,Dtr.shape[-1]+1)] for jj in range(1,len(Dtr)+1)])
        
        A = np.zeros((N,T))+np.nan
        for t in range(T0):
            A[:,t] = logit(Pobstr2[:,t])
        for t in range(T0, T):
            A[:,t] = a0 + a1*A[:,t-1] + a2*A[:,t-2] - np.sum(Dtr2[:,t]*b, axis=1)
        p = sigmoid(A)
        
        Ppte.append(p)
        Epte.append(np.random.binomial(W, p))
    Ppte = np.array(Ppte)
    Epte = np.array(Epte)
    
    Epte = Epte/W
    Ep_tr_sim.append(Epte[:,:-1])
    Ep_te_sim.append(Epte[:,-1])
    E_te.append(Pobstr2[-1])
    D_te.append(Dtr2[-1]*Dmax)
    
    # baseline
    
    # first decide which value to carry forward
    # it should be the (Tcomb-1)-th, but it can be NaN, search backwards until non-NaN
    for tt in range(Tcomb-1,-1,-1):
        if not np.isnan(Pobstr[-1][tt]):
            break
    Epte_baseline = np.random.binomial(W, Pobstr[-1][tt], size=len(Ppte))
    Epte_baseline = Epte_baseline/W
    Epte_baseline = np.array([Epte_baseline]*(T-tt)).T
    Epte_baseline = np.c_[np.array([Pobstr[-1][:tt]]*len(Ppte)), Epte_baseline]
        
    Ep_te_baseline.append(Epte_baseline)
    
    plt.close()
    fig = plt.figure()
    ax1=fig.add_subplot(211)
    Epte3 = np.percentile(Epte, (50,2.5,97.5), axis=0)
    ax1.plot(Epte3[:,-1,:].T,c='r')
    ax1.plot(Pobstr2[-1],c='k')
    ax1.axvline(Tcomb, c='r', ls='--', lw=2)
    ax2=fig.add_subplot(212)
    ax2.plot(Dtr2[-1]*Dmax)
    ax2.axvline(Tcomb, c='r', ls='--', lw=2)
    #plt.show()
    plt.savefig('%s.png'%sid)
    
    with open('results.pickle', 'wb') as ff:
        pickle.dump({'fit_res':fit_res,
                     #'Ep_tr_sim':Ep_tr_sim,
                     'Ep_te_sim':Ep_te_sim,
                     'Ep_te_baseline':Ep_te_baseline,
                     'E_te':E_te,
                     'D_te':D_te,
                     'sids':sids_te}, ff)

