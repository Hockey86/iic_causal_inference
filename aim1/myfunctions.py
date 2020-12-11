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


def patient(path, W):
    window = W
    step   = W
    res = {}
    mat = sio.loadmat(path)

    drugs = mat['drugs_weightnormalized'].toarray().astype(float)
    drugnames = list(map(lambda x: x.strip(), mat['Dnames']))
    window_start_ids = np.arange(0,len(drugs),step)
    drugs_window = np.array([ np.mean(drugs[i:i+window],axis=0) for i in window_start_ids ])
    for i, dn in enumerate(drugnames):
        res[dn] = drugs_window[:,i]
    res['window_start_ids'] = window_start_ids
    
    if 'artifact' in mat:
        artifact = mat['artifact'].flatten().astype(float)
    
    if 'iic' in mat:
        for smooth in ['', '_smooth']:
            if 'iic'+smooth not in mat:
                continue
            iic = mat['iic'+smooth].flatten().astype(float)
            if 'artifact' in mat:
                iic[artifact==1] = np.nan
            nan_ids = np.isnan(iic)
            iic_burden = np.in1d(iic, [1,2,3,4]).astype(float)
            iic_burden[nan_ids] = np.nan
            iic_burden_window = np.array([np.nanmean(iic_burden[i:i+window]) for i in window_start_ids])
            res['iic_burden'+smooth] = iic_burden_window
            sz_burden = (iic==1).astype(float)
            sz_burden[nan_ids] = np.nan
            sz_burden_window = np.array([np.nanmean(sz_burden[i:i+window]) for i in window_start_ids])
            res['sz_burden'+smooth] = sz_burden_window
        
    if 'spike' in mat:
        spike = mat['spike'].flatten().astype(float)
        if 'artifact' in mat:
            spike[artifact==1] = np.nan
        spike_rate_window = np.array([np.nanmean(spike[i:i+window]) for i in window_start_ids])
        res['spike_rate'] = spike_rate_window
    
    if 'spec' in mat:
        spec = mat['spec'].astype(float)
        freq = mat['spec_freq'].flatten().astype(float)
        spec_window = np.array([np.nanmean(spec[i:i+window], axis=0) for i in window_start_ids])
        res['spec'] = spec_window
        res['freq'] = freq
        
    return res


def preprocess(sid, data_dir, PK_K, W, drugs_tostudy, response_tostudy, smooth=False):  # previsously called patient_data
    #fetch the data
    p = patient(os.path.join(data_dir, sid + '.mat'), W)

    #setting up the data
    if smooth:
        Pobs = p[response_tostudy+'_smooth']
    else:
        Pobs = p[response_tostudy]

    #PK
    Ddose = np.array([p[x] for x in drugs_tostudy])
    D = drug_concentration(Ddose, PK_K[drugs_tostudy].values).T

    #cov_tostudy = ['Age']
    C = pd.read_csv(os.path.join(data_dir, 'covariates.csv'))
    Cname = list(C.columns)
    C = C[C.Index==sid].values[0]#[cov_tostudy]
    
    Y = pd.read_csv(os.path.join(data_dir, 'outcomes.csv'))
    Y = Y['DC MRS (modified ranking scale)'][Y.Index==sid].values[0]#[cov_tostudy]
    
    return Pobs, D, drugs_tostudy, C, Cname, Y, p['window_start_ids']#, p['spec'], p['freq']
    
