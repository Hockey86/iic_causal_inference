from itertools import groupby
import os
import pickle
import sys
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
    
    response_tostudy = str(sys.argv[1])# iic_burden_smooth or spike_rate
    outcome_tostudy = 'DC MRS (modified ranking scale)'
    
    ## preprocess data
    
    W = 300
    pmrns = []
    Pobs = []
    D = []
    Ddose = []
    C = []
    Y = []
    window_start_ids = []
    #spec = []
    #freq = []
    for si, sid in enumerate(tqdm(sids)):
        pmrn, Pobs_, D_, Ddose_, Dname, C_, Cname, Y_, ids = preprocess(sid, DATA_DIR, PK_K, W, drugs_tostudy, response_tostudy, outcome_tostudy)
        pmrns.append(pmrn)
        Pobs.append(Pobs_)
        D.append(D_)
        Ddose.append(Ddose_)
        C.append(C_)
        Y.append(Y_)
        window_start_ids.append(ids)
        #spec.append(spec_)
        #freq.append(freq_)
    Y = np.array(Y)
    C = np.array(C)
    #sids = C[:,0].astype(str)
    C = C[:,1:].astype(float)
    Cname = Cname[1:]
    
    # get cluster
    df_cluster = pd.read_csv('Cluster_2000pts_using_C_12clusters.csv')
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
    
    # exclude patients with short signal
    min_len = 12
    keep_ids = np.array([len(D[i])>=min_len for i in range(len(sids))])
    
    # exclude patients with IIC = 0
    #keep_ids &= np.array([np.nanmax(Pobs[i])>0 for i in range(len(sids))])
    # exclude patients with no drug
    #keep_ids &= np.array([~np.all(D[i]==0) for i in range(len(sids))])
    
    keep_ids = np.where(keep_ids)[0]
    pmrns = [pmrns[i] for i in keep_ids]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Ddose = [Ddose[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    window_start_ids = [window_start_ids[i] for i in keep_ids]
    print('%d patients'%len(sids))

    """
    # # remove flat drug at the beginning or end

    #starts = []
    #ends = []
    for i in range(len(sids)):
        d = D[i].sum(axis=1)

        start = 0
        for gi, g in enumerate(groupby(d)):
            j, k = g
            ll = len(list(k))
            if j==0:
                start = ll
            break

        end = 0
        for gi, g in enumerate(groupby(d[::-1])):
            j, k = g
            ll = len(list(k))
            if j==0:
                end = ll
            break
        end = len(d)-end

        Pobs[i] = Pobs[i][start:end]
        D[i] = D[i][start:end]
        Ddose[i] = Ddose[i][start:end]
        window_start_ids[i] = window_start_ids[i][start:end]
        #starts.append(start)
        #ends.append(end)

    keep_ids = np.where([len(D[i])>=min_len for i in range(len(sids))])[0]
    pmrns = [pmrns[i] for i in keep_ids]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Ddose = [Ddose[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    window_start_ids = [window_start_ids[i] for i in keep_ids]
    print('%d patients'%len(sids))
    """
    
    ## clip NaN drug at the beginning or end
    ## remove patients with NaN drug in the middle
    exclude_ids = []
    for i in range(len(sids)):
        d = D[i].sum(axis=1)
        nan_mask = np.isnan(d)
        if np.sum(nan_mask)==0:
            continue
        
        d[np.isnan(d)] = -999

        start = 0
        for gi, g in enumerate(groupby(d)):
            j, k = g
            ll = len(list(k))
            if j==-999:
                start = ll
            break

        end = 0
        for gi, g in enumerate(groupby(d[::-1])):
            j, k = g
            ll = len(list(k))
            if j==-999:
                end = ll
            break
        end = len(d)-end
        
        cc = 0
        for gi, g in enumerate(groupby(d)):
            j, k = g
            ll = len(list(k))
            if j==-999 and cc>0 and cc+ll<len(d):
                exclude_ids.append(i)
                break
            cc += ll

        Pobs[i] = Pobs[i][start:end]
        D[i] = D[i][start:end]
        Ddose[i] = Ddose[i][start:end]
        window_start_ids[i] = window_start_ids[i][start:end]

    keep_ids = np.where([len(D[i])>=min_len and i not in exclude_ids for i in range(len(sids))])[0]
    pmrns = [pmrns[i] for i in keep_ids]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Ddose = [Ddose[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    window_start_ids = [window_start_ids[i] for i in keep_ids]
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
    pmrns = [pmrns[i] for i in keep_ids]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Ddose = [Ddose[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    window_start_ids = [window_start_ids[i] for i in keep_ids]
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
        Ddose[i] = Ddose[i][t:]
        window_start_ids[i] = window_start_ids[i][t:]

    keep_ids = np.where([len(D[i])>=min_len for i in range(len(sids))])[0]
    pmrns = [pmrns[i] for i in keep_ids]
    sids = [sids[i] for i in keep_ids]
    Pobs = [Pobs[i] for i in keep_ids]
    D = [D[i] for i in keep_ids]
    Ddose = [Ddose[i] for i in keep_ids]
    Y = Y[keep_ids]
    C = C[keep_ids]
    cluster = LabelEncoder().fit_transform(cluster[keep_ids])
    window_start_ids = [window_start_ids[i] for i in keep_ids]
    print('%d patients'%len(sids))
    
    assert all([np.all(~np.isnan(x)) for x in D])
    assert all([np.all(~np.isnan(x)) for x in Ddose])
    #assert all([np.all(~np.isnan(Pobs)) for x in Pobs])
    
    output_path = f'data_to_fit_CNNIIC_{response_tostudy}.pickle'
    with open(output_path, 'wb') as f:
        pickle.dump({'W':W, 'window_start_ids':window_start_ids,
                     'D':D, 'Ddose':Ddose, 'Dname':Dname,
                     'Pobs':Pobs, 'Pname':response_tostudy,
                     'C':C, 'Cname':Cname,
                     'Y':Y, 'Yname':outcome_tostudy,
                     'cluster':cluster, 'sids':sids, 'pseudoMRNs':pmrns,}, f)

