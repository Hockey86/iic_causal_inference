from itertools import groupby
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from myfunctions import preprocess, get_pk_k


if __name__=='__main__':

    DATA_DIR = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'
    sids = [x.replace('.mat','') for x in os.listdir(DATA_DIR) if x.endswith('.mat')]
    sids = sorted(sids, key=lambda x:int(x[len('sid'):]))
    
    # # define PK time constants for each drug
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam',
                    'pentobarbital','phenobarbital',# 'phenytoin',
                    'propofol', 'valproate']
    PK_K = get_pk_k()
    
    #response_tostudy = 'iic_burden'
    response_tostudy = 'spike_rate'
    
    ## preprocess data
    
    W = 300
    Pobs = []
    D = []
    C = []
    Y = []
    #spec = []
    #freq = []
    for sid in tqdm(sids):
        Pobs_, Pname, D_, Dname, C_, Cname, Y_ = preprocess(sid, DATA_DIR, PK_K, W, drugs_tostudy, response_tostudy)
        Pobs.append(Pobs_)
        D.append(D_)
        C.append(C_)
        Y.append(Y_)
        #spec.append(spec_)
        #freq.append(freq_)
    import pdb;pdb.set_trace()
    Y = np.array(Y)
    C = np.array(C)
    #sids = C[:,0].astype(str)
    C = C[:,1:].astype(float)
    Cname = Cname[1:]

    # get cluster
    df_cluster = pd.read_csv('Cluster_2000pts_using_C_2clusters.csv')
    sids2 = list(df_cluster.Index)
    cluster = []
    C2 = (C-np.nanmean(C,axis=0))/np.nanstd(C,axis=0)
    for si, sid in enumerate(sids):
        if sid in sids2:
            cluster.append(df_cluster['12_Cluster'].iloc[sids2.index(sid)])
        else:
            dists = np.nanmean((C2 - C2[si])**2, axis=1)
            rank = np.argsort(dists)
            closest_id = rank[rank!=si][0]
            cluster.append(df_cluster['12_Cluster'].iloc[closest_id])
    cluster = np.array(cluster)

    print('%d patients'%len(sids))
    
    # exclude patients with IIC = 0
    keep_ids = np.array([np.nanmax(Pobs[i])>0 for i in range(len(sids))])
    # exclude patients with short signal
    min_len = 30
    keep_ids &= np.array([len(D[i])>=min_len for i in range(len(sids))])
    # exclude patients with missing dose
    keep_ids &= np.array([~np.any(np.isnan(D[i])) for i in range(len(sids))])
    # exclude patients with no drug
    keep_ids &= np.array([~np.all(D[i]==0) for i in range(len(sids))])
    
    keep_ids = np.where(keep_ids)[0]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    print('%d patients'%len(sids))

    # # remove flat drug at the beginning or end

    #starts = []
    #ends = []
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
        #starts.append(start)
        #ends.append(end)

    keep_ids = np.where([len(D[i])>=min_len for i in range(len(sids))])[0]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    print('%d patients'%len(sids))
    
    # # remove subjects with long continuous NaN in Pobs
    thres = 0.3
    keep_ids = []
    for i in range(len(sids)):
        p = np.array(Pobs[i])
        p[np.isnan(p)] = -999

        good = True
        for j, k in groupby(p):
            if j==-999:
                ll = len(list(k))
                if ll/len(p)>thres:
                    good = False
                    break
        if good:
            keep_ids.append(i)
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    print('%d patients'%len(sids))
    
    # # make sure the first T0 points are not NaN to initialize the model
    
    T0 = 6
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

    keep_ids = np.where([len(D[i])>=min_len for i in range(len(sids))])[0]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    print('%d patients'%len(sids))
    
    import pdb;pdb.set_trace()
    with open('data_to_fit_CNNIIC_%s.pickle'%response_tostudy, 'wb') as f:
        pickle.dump({'W':W, 'D':D, 'Dname':Dname,
                     'Pobs':Pobs, 'Pname':Pname,
                     'cluster':cluster, 'sids':sids,
                     'C':C, 'Cname':Cname, 'Y':Y}, f)

