from collections import Counter
from itertools import groupby
import os
import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from hmmlearn.hmm import MultinomialHMM


def get_startprob(seqs, n_components, pseudocount=1):
    startprob = np.zeros(n_components)+pseudocount
    for seq in seqs:
        nan_indicator = np.isnan(seq)
        cc = 0
        for isnan, l in groupby(nan_indicator):
            ll = len(list(l))
            if not isnan:
                first_value = int(seq[cc])
                startprob[first_value] += 1
                break
            cc += ll
        
    startprob = startprob/startprob.sum()
    return startprob


def get_transmat(seqs, n_components, pseudocount=1):
    transmat = np.zeros((n_components, n_components))+pseudocount
    for seq in seqs:
        nan_indicator = np.isnan(seq)
        cc = 0
        for isnan, l in groupby(nan_indicator):
            ll = len(list(l))
            if not isnan:
                seq_ = seq[cc:cc+ll].astype(int)
                pair_counter = Counter([(seq_[i], seq_[i+1]) for i in range(len(seq_)-1)])
                for kk, vv in pair_counter.items():
                    transmat[kk[0],kk[1]] += vv
            cc += ll
    transmat = transmat/transmat.sum(axis=1, keepdims=True)
    return transmat


def get_emissionprob(seqs, seqs_obs, n_components, pseudocount=1):
    emissionprob = np.zeros((n_components, n_components))+pseudocount
    for seq, seq_obs in zip(seqs, seqs_obs):
        nan_indicator = np.isnan(seq) | np.isnan(seq_obs)
        cc = 0
        for isnan, l in groupby(nan_indicator):
            ll = len(list(l))
            if not isnan:
                seq_ = seq[cc:cc+ll].astype(int)
                seq_obs_ = seq_obs[cc:cc+ll].astype(int)
                pair_counter = Counter([(seq_[i], seq_obs_[i]) for i in range(len(seq_))])
                for kk, vv in pair_counter.items():
                    emissionprob[kk[0],kk[1]] += vv
            cc += ll
    emissionprob = emissionprob/emissionprob.sum(axis=1, keepdims=True)
    return emissionprob
    

def hmmpredict(hmm, seq):
    res = np.zeros_like(seq)+np.nan
    nan_indicator = np.isnan(seq)
    cc = 0
    for isnan, l in groupby(nan_indicator):
        ll = len(list(l))
        if not isnan:
            seq_ = seq[cc:cc+ll].astype(int)
            res[cc:cc+ll] = hmm.predict(seq_.reshape(-1,1))
        cc += ll
    return res


if __name__=='__main__':
    random_state = 2020
    human_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
    cnn_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'

    human_iic_sids = sorted([x.replace('.mat','') for x in os.listdir(human_iic_dir) if x.endswith('.mat')], key=lambda x:int(x[len('sid'):]))
    cnn_iic_sids = sorted([x.replace('.mat','') for x in os.listdir(cnn_iic_dir) if x.endswith('.mat')], key=lambda x:int(x[len('sid'):]))

    common_sids = sorted(set(human_iic_sids)&set(cnn_iic_sids), key=lambda x:int(x[len('sid'):]))
    human_iics = []
    cnn_iics = []
    artifacts = []
    
    labels_txt = ['Other','Sz','LPD','GPD','LRDA','GRDA']
    n_components = len(labels_txt)
    for sid in tqdm(common_sids):
        human_iic_mat = sio.loadmat(os.path.join(human_iic_dir, sid+'.mat'))
        cnn_iic_mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
        artifact = human_iic_mat['artifact'].flatten()
        human_iic = human_iic_mat['human_iic'].flatten()
        cnn_iic = cnn_iic_mat['iic'].flatten()
        
        # do not make artifact to nan here,
        # because it creats many short segments,
        # hard for HMM to predict since it restarts once it meets nan
        # also we want to learn dynamics during artifact
        #human_iic[artifact==1] = np.nan
        
        len_ = min([len(human_iic), len(cnn_iic), len(artifact)])
        human_iic = human_iic[:len_]
        cnn_iic = cnn_iic[:len_]
        artifact = artifact[:len_]
        
        human_iics.append(human_iic)
        cnn_iics.append(cnn_iic)
        artifacts.append(artifact)
        
    # HMM smoothing
    hmm = MultinomialHMM(n_components=n_components, random_state=random_state)
    hmm.startprob_ = get_startprob(human_iics, n_components)
    hmm.transmat_ = get_transmat(human_iics, n_components)
    hmm.emissionprob_ = get_emissionprob(human_iics, cnn_iics, n_components)
    
    # try smoothing levels by adding to diagonal of transmat
    all_human_iic = np.concatenate(human_iics)
    #all_cnn_iic = np.concatenate(cnn_iics)
    all_artifacts = np.concatenate(artifacts)
    all_human_iic[all_artifacts==1] = np.nan
    #all_cnn_iic[all_artifacts==1] = np.nan
    
    """
    transmat_ = np.array(hmm.transmat_, copy=True)
    kappas = []
    smoothing_levels = [0,0.1,0.2,0.3,0.4,0.5]
    for sl in smoothing_levels:
        hmm.transmat_ = transmat_ + np.diag(np.zeros(n_components)+sl)
        hmm.transmat_ = hmm.transmat_ / hmm.transmat_.sum(axis=1, keepdims=True)
        cnn_iics_smooth = [hmmpredict(hmm, cnn_iics[si]) for si, sid in enumerate(common_sids)]
        all_cnn_iic_smooth = np.concatenate(cnn_iics_smooth)
        
        # make artifact to nan here after making prediction
        all_cnn_iic_smooth[all_artifacts==1] = np.nan
        
        ids = (~np.isnan(all_human_iic))&(~np.isnan(all_cnn_iic_smooth))
        kappas.append( cohen_kappa_score(all_human_iic[ids], all_cnn_iic_smooth[ids]) )
    best_smoothing_level = smoothing_levels[np.argmax(kappas)]
    best_kappa = np.max(kappas)
    
    hmm_final = MultinomialHMM(n_components=n_components, random_state=random_state)
    hmm_final.startprob_ = hmm.startprob_
    hmm_final.transmat_ = transmat_ + np.diag(best_smoothing_level)
    hmm_final.emissionprob_ = hmm.emissionprob_
    """
    best_smoothing_level = 0
    best_kappa = 0.40273290817848006 # [0.40273290817848006, 0.40247703814739666, 0.4029145614873192, 0.4026801688901007, 0.4024150213440768, 0.4024449937179385]
    hmm_final = hmm
    
    import pdb;pdb.set_trace()
    with open('hmm_smoother_model.pickle', 'wb') as ff:
        pickle.dump({'hmm':hmm_final,
                     'best_smoothing_level':best_smoothing_level,
                     'best_kappa':best_kappa}, ff)
        
    # predict IIC smoothed
    for si, sid in enumerate(tqdm(cnn_iic_sids)):
        path = os.path.join(cnn_iic_dir, sid+'.mat')
        cnn_iic_mat = sio.loadmat(path)
        cnn_iic_mat['iic_smooth'] = hmmpredict(hmm_final,cnn_iic_mat['iic'].flatten())
        sio.savemat(path, cnn_iic_mat)
