import datetime
import os
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


def generate_Y(outcomes):
    col = [x for x in outcomes.columns if 'mrs' in x.lower()]
    assert len(col)==1
    res = outcomes[col[0]].copy()
    res.loc[np.where(res=='na')[0]] = np.nan
    return res.astype(float).values, col


def generate_C_features(covs):

    # delete columns with many nan
    bad_col_ids = list(covs.columns[np.mean(np.isnan(covs.values), axis=0)>0.1])
    print('\n%d variables in C are removed due to > 10%% nan: %s'%(len(bad_col_ids), bad_col_ids))
    covs = covs.drop(bad_col_ids, axis=1)

    # delete not useful columns
    not_useful_cols = ['iGCS = T?', 'iGCS-E', 'iGCS-V', 'iGCS-M',  'iGCS actual scores',
                       #'neuro_dx_none', 'prim_dx_none', 'sz_dx_none',
                       'Worst GCS in 1st 24', 'Worst GCS Intubation status']#, 'Weight', 'BOLT N0=0 Yes=1']
    print('%d variables in C are removed due to not useful: %s'%(len(not_useful_cols), not_useful_cols))
    covs = covs.drop(not_useful_cols, axis=1)

    # delete correlated columns
    correlated_cols = ['iGCS-Total',
                       'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)',
                       'iMV  (initial (on admission) mechanical ventilation)',
                       'diastolic BP',]
    #                   'External Ventricular Drain (EVD)',
    #                   'elevated ICP=more than 20 (either on admission or in hospital course)   QPID'
    #                   ]
    print('%d variables in C are removed due to correlated: %s'%(len(correlated_cols), correlated_cols))
    covs = covs.drop(correlated_cols, axis=1)
    
    """
    # find correlated covs
    cc = np.array([len(set(covs[col])) for col in covs.columns])
    bin_ids = np.where(cc==2)[0]
    cont_ids = np.where(cc!=2)[0]
    thres = 0.5
    corr_pairs = []
    for i, j in combinations(range(len(covs.columns)), 2):
        xi = covs[covs.columns[i]].values
        xj = covs[covs.columns[j]].values
        ids = (~np.isnan(xi))&(~np.isnan(xj))
        xi = xi[ids]
        xj = xj[ids]
        if i in bin_ids and j in bin_ids:  # within binary covs
            corr = cohen_kappa_score(xi, xj)
        elif i in cont_ids and j in cont_ids:  # within continuous covs
            corr = spearmanr(xi, xj)[0]
        else:
            continue
        if np.abs(corr)>=thres:
            corr_pairs.append( (covs.columns[i], covs.columns[j], corr) )
    assert len(corr_pairs)==0
    """
    return covs.values, np.array(list(covs.columns))


def PK(d, length, a, b, c, dt=0.1):
    x = [[0,0]]
    for t in range(1,length):
        x.append([
            x[-1][0] + dt*(- (a+b)*x[-1][0] + c*x[-1][1] + 10*d[t-1]),
            x[-1][1] + dt*(b*x[-1][0]-c*x[-1][1])
        ])
    return np.array(x)[:,1]


if __name__=='__main__':
    
    sids = ['sid36', 'sid39', 'sid56', 'sid297', 'sid327', 'sid385',
       'sid395', 'sid400', 'sid403', 'sid406', 'sid424', 'sid450',
       'sid456', 'sid490', 'sid512', 'sid551', 'sid557', 'sid575',
       'sid988', 'sid1016', 'sid1025', 'sid1034', 'sid1038', 'sid1039',
       'sid1055', 'sid1056', 'sid1063', 'sid1337', 'sid1897', 'sid1913',
       'sid1915', 'sid1916', 'sid1917', 'sid1926', 'sid1928', 'sid1956',
       'sid1966']
       
    tostudy_Dnames = [
            'levetiracetam', 'lacosamide',
            'fosphenytoin',# 'phenytoin',# these are the same
            'valproate',# 'divalproex',# these are the same
            'propofol', 'midazolam']#, 'pentobarbital']
       
    ## generate C
    covs = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh/covariates.csv')
    assert covs.Index.tolist()==sids
    C, Cnames = generate_C_features(covs.drop(columns='Index'))
    
    ## generate Y
    outcomes = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh/outcomes.csv')
    assert outcomes.Index.tolist()==sids
    Y, Ynames = generate_Y(outcomes.drop(columns='Index'))

    """
    ## remove nan in Y and C
    goodids = np.where(~( np.isnan(Y) | np.any(np.isnan(C),axis=1) ))[0]
    if len(goodids)<len(Y):
        print('%d/%d (%.1f%%) patients are kept. Others are removed due to NaN\'s in Y and C'%(len(goodids), len(Y), len(goodids)*100./len(Y)))
        Y = Y[goodids]
        sids = sids[goodids]

        C = C[goodids]
        goodids = np.std(C, axis=0)>0
        C = C[:,goodids]
        Cnames = Cnames[goodids]
    """

        
    data_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
    window_times = [1*60, 10*60, 3600, 3*3600, 6*3600]#, 12*3600, 24*3600  # [s]
    num_windows = [10,6,4,4,2]#,2,2
    
    
    # a/b ^, clearance^
    # c ^, clearance^
    As = {'levetiracetam':1,
          'lacosamide':1,
          'fosphenytoin':1,
          'valproate':1,
          'propofol':5,
          'midazolam':5}
          
    Bs = {'levetiracetam':10,
          'lacosamide':10,
          'fosphenytoin':10,
          'valproate':10,
          'propofol':1,
          'midazolam':1}
          
    Cs = {'levetiracetam':0.01,
          'lacosamide':0.01,
          'fosphenytoin':0.01,
          'valproate':0.01,
          'propofol':1,
          'midazolam':1}
        
    X = []
    y = []
    info = []
    
    # for each patient
    for si, sid in enumerate(tqdm(sids)):
        res = sio.loadmat(os.path.join(data_dir, sid+'.mat'))
        spec = res['spec']
        spec[np.isinf(spec)] = np.nan
        freq = res['spec_freq'].flatten()
        spike = res['spike'].flatten()
        start_time = res['start_time'][0].strip()
        human_label = res['human_iic'].flatten()
        artifact_indicator = res['artifact'].flatten()
        
        human_label[artifact_indicator==1] = np.nan
        spike[artifact_indicator==1] = np.nan
        
        Dnames = [x.strip() for x in res['Dnames']]
        ids = [Dnames.index(x) for x in tostudy_Dnames]
        drugs = res['drugs_weightnormalized'][:,ids]
        u = np.array([PK(drugs[:,k], len(drugs), As[tostudy_Dnames[k]], Bs[tostudy_Dnames[k]], Cs[tostudy_Dnames[k]], dt=0.1) for k in range(len(tostudy_Dnames))]).T
        
        # for each drug
        for di, dn in enumerate(tostudy_Dnames):
            this_drug = drugs[:,di]
            
            # get drug dose change points
            change_ids = np.where((np.diff(this_drug)>0)|(np.diff(this_drug<0)))[0]
            
            # for each change point
            for ci in change_ids:
                current_time = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')+datetime.timedelta(seconds=int(ci)*2)
                current_time_txt = datetime.datetime.strftime(current_time, '%Y-%m-%d %H-%M-%S')
            
                """
                window_time = window_times[-1]
                window_size = int(window_time)//2
                
                start = max(0, ci-window_size)
                end = min(len(this_drug), ci+window_size)
                
                artifact = artifact_indicator[start:end]
                spec_ = spec[start:end]
                if np.all(np.isnan(spec_[:ci-start])):
                    continue   # if no EEG at all (gap) before drug change point, ignore
                if np.all(np.isnan(spec_[ci-start:])):
                    continue   # if no EEG at all (gap) after drug change point, ignore
                human_label_ = human_label[start:end]
                #nanids = np.all(np.isnan(res['iic'][start:end]), axis=1)
                #iic_cnn_prediction = np.argmax(res['iic'][start:end], axis=1).astype(float)
                #iic_cnn_prediction[nanids] = np.nan
                """
                
                # for each time scale
                
                sz_burdens = []
                iic_burdens = []
                spike_rates = []
                drug_concentrations = []
                for wi in range(len(window_times)):
                    window_size = int(window_times[wi])//2
                
                    start = max(0, ci-window_size)
                    end = min(len(this_drug), ci+window_size)
                    start_id = np.r_[ci-np.arange(0, ci-start-window_size+1, window_size)[::-1]-window_size,
                                      ci+np.arange(0, end-ci-window_size+1, window_size)]
                    if len(start_id)!=2:
                        continue
                        
                    # segment human_label into windows
                    human_label_segs = human_label.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+window_size), start_id))][0]
                    # directly find sz (1) pattern will make NaN look like False,
                    # so first find where are NaN's
                    nanids = np.isnan(human_label_segs)
                    sz_burden = (human_label_segs==1).astype(float)
                    # after finding sz, then set to NaN where it is originally NaN
                    sz_burden[nanids] = np.nan
                    # then take the mean, mean of binary array = % of 1's
                    sz_burden = np.nanmean(sz_burden, axis=1)*100
                    sz_burdens.append(sz_burden)
                    
                    iic_burden = ((human_label_segs>=1) & (human_label_segs<=4)).astype(float)
                    iic_burden[nanids] = np.nan
                    iic_burden = np.nanmean(iic_burden, axis=1)*100
                    iic_burdens.append(iic_burden)
                    
                    # segment spike_rate into windows
                    # spike_rate.shape = (#2s-window,)
                    spike_rate = spike.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+window_size), start_id))][0]
                    # spike_rate.shape = (#window, window_size)
                    nanids = np.all(np.isnan(spike_rate), axis=1)
                    spike_rate = np.nansum(spike_rate, axis=1)/window_size*60
                    spike_rate[nanids] = np.nan
                    spike_rates.append(spike_rate)
                    
                    u_segs = u.T[:, list(map(lambda x:np.arange(x,x+window_size), start_id))].transpose(1,0,2)
                    u_segs = np.nansum(u_segs, axis=2)/3600.
                    drug_concentrations.append(u_segs)
                    
                    X.append(np.r_[u_segs[1]-u_segs[0], u_segs[0], sz_burden[0], iic_burden[0], spike_rate[0], C[si]])
                    y.append([sz_burden[1], iic_burden[1], spike_rate[1]])
                    info.append([sid, dn, ci, window_times[wi]])
        
    Xnames = ['diff_'+x for x in tostudy_Dnames] + ['baseline_'+x for x in tostudy_Dnames] + ['baseline_Sz_burden', 'baseline_IIC_burden', 'baseline_spike_rate'] + list(Cnames)
    ynames = ['new_sz_burden', 'new_iic_burden', 'new_spike_rate']
    infonames = ['sid', 'drug_name', 'change_point_id', 'window_time_second']
    X2 = pd.DataFrame(data=np.array(X, dtype=float), columns=Xnames)
    y2 = pd.DataFrame(data=np.array(y, dtype=float), columns=ynames)
    info2 = pd.DataFrame(data=np.array(info, dtype=object), columns=infonames)
    
    ## save

    X2.to_csv('X.csv', index=False)
    y2.to_csv('y.csv', index=False)
    info2.to_csv('info.csv', index=False)

