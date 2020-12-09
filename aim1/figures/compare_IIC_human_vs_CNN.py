from collections import Counter
from itertools import groupby
import os
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from hmmlearn.hmm import MultinomialHMM
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


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

    human_iic_sids = [x.replace('.mat','') for x in os.listdir(human_iic_dir) if x.endswith('.mat')]
    cnn_iic_sids = [x.replace('.mat','') for x in os.listdir(cnn_iic_dir) if x.endswith('.mat')]

    common_sids = sorted(set(human_iic_sids)&set(cnn_iic_sids), key=lambda x:int(x[len('sid'):]))
    human_iics = []
    cnn_iics = []
    specs = []
    artifacts = []
    
    labels_txt = ['Other','Sz','LPD','GPD','LRDA','GRDA']
    for sid in tqdm(common_sids):
        human_iic_mat = sio.loadmat(os.path.join(human_iic_dir, sid+'.mat'))
        cnn_iic_mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
        artifact = human_iic_mat['artifact'].flatten()
        spec = human_iic_mat['spec']
        freq = human_iic_mat['spec_freq'].flatten()
        human_iic = human_iic_mat['human_iic'].flatten()
        cnn_iic = cnn_iic_mat['iic'].flatten()
        
        # do not make artifact to nan here,
        # because it creats many short segments,
        # hard for HMM to predict since it restarts once it meets nan
        # also we want to learn dynamics during artifact
        #human_iic[artifact==1] = np.nan
        
        len_ = min([len(human_iic), len(cnn_iic), len(spec), len(artifact)])
        human_iic = human_iic[:len_]
        cnn_iic = cnn_iic[:len_]
        spec = spec[:len_]
        artifact = artifact[:len_]
        
        human_iics.append(human_iic)
        cnn_iics.append(cnn_iic)
        specs.append(spec)
        artifacts.append(artifact)
        
    # HMM smoothing
    n_components = len(labels_txt)
    hmm = MultinomialHMM(n_components=n_components, random_state=random_state)
    hmm.startprob_ = get_startprob(human_iics, n_components)
    hmm.transmat_ = get_transmat(human_iics, n_components)
    hmm.emissionprob_ = get_emissionprob(human_iics, cnn_iics, n_components)
        
    vmin = -20
    vmax = 20
    cnn_iics2 = []
    for si, sid in enumerate(tqdm(common_sids)):
        human_iic = human_iics[si]
        cnn_iic = cnn_iics[si]
        spec = specs[si]
        cnn_iic2 = hmmpredict(hmm, cnn_iic)
        
        # make artifact to nan here after making prediction
        cnn_iic2[artifacts[si]==1] = np.nan
        cnn_iics[si][artifacts[si]==1] = np.nan
        human_iics[si][artifacts[si]==1] = np.nan
        
        cnn_iics2.append(cnn_iic2)
        
        continue
        tt = np.arange(len(human_iic))*2/3600
        plt.close()
        fig = plt.figure(figsize=(12,10))
        ax1 = fig.add_subplot(411)
        ax1.imshow(spec.T, aspect='auto', origin='lower',cmap='jet',vmin=vmin, vmax=vmax,
                    extent=(tt.min(), tt.max(), freq.min(), freq.max()))
        
        ax2 = fig.add_subplot(412, sharex=ax1)
        ax2.plot(tt, human_iic, c='k')
        ax2.set_yticks([-1]+list(range(n_components))+[n_components])
        ax2.set_yticklabels(['']+labels_txt+[''])
        ax2.set_ylabel('IIC from human')
        #ax2.set_xlabel('Time (hour)')
        ax2.set_xlim([tt.min(), tt.max()])
        ax2.set_ylim([-1,n_components])
        sns.despine()
        
        ax3 = fig.add_subplot(413, sharex=ax1)
        ax3.plot(tt, cnn_iic, c='m')
        ax3.set_yticks([-1]+list(range(n_components))+[n_components])
        ax3.set_yticklabels(['']+labels_txt+[''])
        ax3.set_ylabel('IIC from CNN')
        #ax3.set_xlabel('Time (hour)')
        ax3.set_xlim([tt.min(), tt.max()])
        ax3.set_ylim([-1,n_components])
        sns.despine()
        
        ax4 = fig.add_subplot(414, sharex=ax1)
        ax4.plot(tt, cnn_iic2, c='b')
        ax4.set_yticks([-1]+list(range(n_components))+[n_components])
        ax4.set_yticklabels(['']+labels_txt+[''])
        ax4.set_ylabel('IIC from CNN + HMM')
        ax4.set_xlabel('Time (hour)')
        ax4.set_xlim([tt.min(), tt.max()])
        ax4.set_ylim([-1,n_components])
        sns.despine()
        
        plt.tight_layout()
        #plt.show()
        plt.savefig('human_vs_CNN_IIC/%s.png'%sid)
        


    all_human_iic = np.concatenate(human_iics)
    all_cnn_iic = np.concatenate(cnn_iics)
    all_cnn_iic2 = np.concatenate(cnn_iics2)
    
    ids = (~np.isnan(all_human_iic))&(~np.isnan(all_cnn_iic))&(~np.isnan(all_cnn_iic2))
    all_human_iic = all_human_iic[ids]
    all_cnn_iic = all_cnn_iic[ids]
    all_cnn_iic2 = all_cnn_iic2[ids]

    cf = confusion_matrix(all_human_iic,all_cnn_iic)
    cf_prob = cf/cf.sum(axis=1,keepdims=True)
    kappa = cohen_kappa_score(all_human_iic,all_cnn_iic)
    print(kappa)  # 0.39

    cf2 = confusion_matrix(all_human_iic,all_cnn_iic2)
    cf_prob2 = cf2/cf2.sum(axis=1,keepdims=True)
    kappa2 = cohen_kappa_score(all_human_iic,all_cnn_iic2)
    print(kappa2)  # 0.40

    plt.close()
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111)
    sns.heatmap(np.flipud(cf_prob),vmin=0,vmax=1,cmap='Blues',annot=True,square=True,ax=ax,
                xticklabels=labels_txt,
                yticklabels=labels_txt[::-1],)
    ax.set_xlabel('CNN Predicted')
    ax.set_ylabel('Human Annotated')
    plt.tight_layout()
    #plt.show()
    plt.savefig('CNN_vs_human_IIC_cf.png', bbox_inches='tight', pad_inches=0.05)

    plt.close()
    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111)
    sns.heatmap(np.flipud(cf_prob2),vmin=0,vmax=1,cmap='Blues',annot=True,square=True,ax=ax,
                xticklabels=labels_txt,
                yticklabels=labels_txt[::-1],)
    ax.set_xlabel('CNN Predicted')
    ax.set_ylabel('Human Annotated')
    plt.tight_layout()
    #plt.show()
    plt.savefig('CNN_vs_human_IIC_cf_smoothed.png', bbox_inches='tight', pad_inches=0.05)

