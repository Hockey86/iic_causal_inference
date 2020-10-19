import os
import numpy as np
import scipy.io as sio
import pandas as pd


def get_pk_k():
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
    PK_K = halflife.loc['k']
    return PK_K
    
def drug_concentration(d_ts,k):
    """
    d_ts.shape = (#drug, T)
    """
    k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
    conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
    conc = conc[:,:d_ts.shape[1]]
    return conc


# # data functions


def patient(path, W):
    window = W
    step   = W
    
    #if '.mat' in path:
    mat = sio.loadmat(path)
    human_iic = mat['human_iic'].flatten().astype(float)
    spike = mat['spike'].flatten().astype(float)
    drugs = mat['drugs_weightnormalized'].astype(float)
    artifact = mat['artifact'].flatten().astype(float)
    freq = mat['spec_freq'].flatten().astype(float)
    spec = mat['spec'].astype(float)
    human_iic[artifact==1] = np.nan
    spike[artifact==1] = np.nan

    drugnames = list(map(lambda x: x.strip(), mat['Dnames']))
    drugs_window = np.array([ np.mean(drugs[i:i+window],axis=0) for i in range(0,len(drugs),step) ])

    sz_burden = (human_iic==1).astype(float)
    sz_burden[np.isnan(human_iic)] = np.nan
    sz_burden_window = np.array([np.nanmean(sz_burden[i:i+window]) for i in range(0, len(sz_burden),step)])

    iic_burden = np.in1d(human_iic, [1,2,3,4]).astype(float)
    iic_burden[np.isnan(human_iic)] = np.nan
    iic_burden_window = np.array([np.nanmean(iic_burden[i:i+window]) for i in range(0, len(iic_burden),step)])

    spike_rate_window = np.array([np.nanmean(spike[i:i+window]) for i in range(0, len(spike),step)])

    spec_window = np.array([np.nanmean(spec[i:i+window], axis=0) for i in range(0, len(spec),step)])
    
    res = {'sz_burden': sz_burden_window,
           'iic_burden': iic_burden_window,
           'spike_rate': spike_rate_window,
           'spec': spec_window,
           'freq': freq}
    for i, dn in enumerate(drugnames):
        res[dn] = drugs_window[:,i]
    return res


# In[6]:


def preprocess(sid, data_dir, PK_K, W, drugs_tostudy):  # previsously called patient_data

    #fetch the data
    p = patient(os.path.join(data_dir, sid + '.mat'), W)

    #setting up the data
    response_tostudy = 'iic_burden'
    Pobs = p[response_tostudy]

    #PK
    Ddose = np.array([p[x] for x in drugs_tostudy])
    D = drug_concentration(Ddose, PK_K[drugs_tostudy].values).T

    #cov_tostudy = ['Age']
    C = pd.read_csv(os.path.join(data_dir, 'covariates.csv'))
    Cname = list(C.columns)
    C = C[C.Index==sid].iloc[0]#[cov_tostudy]
    
    return Pobs, response_tostudy, D, drugs_tostudy, C, Cname, p['spec'], p['freq']
