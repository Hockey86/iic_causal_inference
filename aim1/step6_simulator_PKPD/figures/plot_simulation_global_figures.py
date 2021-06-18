import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import scipy.io as sio
from scipy.special import expit as sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import seaborn
sys.path.insert(0, '..')
from simulator import *


data_type = 'CNNIIC'
response_tostudy = 'iic_burden_smooth'
#response_tostudy = 'spike_rate'

AR_p = 2
MA_q = 6
max_iter = 1000
Dnames = ['lacosamide', 'levetiracetam', 'midazolam',
            'pentobarbital','phenobarbital',# 'phenytoin',
            'propofol', 'valproate']
ND = len(Dnames)
#models = ['lognormal']#, 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
models = ['cauchy_expit_brandon_ARMA']
random_state = 2020
cnn_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'
    
with open(f'../../data_to_fit_{data_type}_iic_burden_smooth+spike_rate.pickle', 'rb') as f:
    res = pickle.load(f)
window_start_ids = res['window_start_ids']
sids_ = res['sids']
Pobs = res['Pobs'][response_tostudy]
        
halflife_dict = {
    'lacosamide':[66],     #  11h, (5-15h)
    'levetiracetam':[48],  #  8h
    'midazolam':[15],      #  2.5h
    'pentobarbital':[195], # 32.5h (15-50h)
    'phenobarbital':[474], # 79h
    'phenytoin':[147],     # 24.5h (7-42h)
    'propofol':[2],        # 20minutes (3-12h after long time) (needs 3 differential equations)
    'valproate':[96]       # 16h
}

for model in models:
    print(model)
    
    figure_dir = f'simulation_global_figures/{data_type}_{response_tostudy}_{model}{AR_p},{MA_q}'
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    with open(f'../results_{response_tostudy}/results_{data_type}_{response_tostudy}_{model}{AR_p},{MA_q}_iter{max_iter}.pickle', 'rb') as ff:
        res  = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
    assert np.all(sids_==sids)
    
    # recover drug dose
    for i in range(len(Ddose)):
        Ddose[i] = Ddose[i]*Ddose_max
        
    stan_path = f'../stan_models/model_{model}.stan'
    model_path = f'../results_{response_tostudy}/model_fit_{data_type}_{response_tostudy}_{model}{AR_p},{MA_q}_iter{max_iter}.pkl'
    simulator = Simulator(stan_path, W, T0=[AR_p, MA_q], max_iter=max_iter, random_state=random_state)
    simulator.load_model(model_path)
    # get vmin and vmax
    """
    sids_subset = np.random.choice(sids, 50)
    specs_subset = []
    for sid in sids_subset:
        mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
        spec = mat['spec'].flatten()
        spec[np.isinf(spec)] = np.nan
        spec = spec[~np.isnan(spec)]
        specs_subset.append(spec)
    specs_subset = np.concatenate(specs_subset)
    #vmin, vmax = np.percentile(specs_subset, (5, 95))
    """
    vmin = -20
    vmax = 20
    if response_tostudy == 'spike_rate':
        yscale = 60
        ylim = [-1,61]
    elif response_tostudy.startswith('iic_burden'):
        yscale = 100
        ylim = [-2,102]
    
    # save correlation
    corrs = []
    ps = []
    for si, sid in enumerate(tqdm(sids)):
        P1 = Pobs[si]
        P2 = np.mean(Psim[si], axis=0)
        ids = (~np.isnan(P1)) & (~np.isnan(P2))
        P1 = P1[ids]
        P2 = P2[ids]
        corr, p = spearmanr(P1,P2)
        corrs.append(corr)
        ps.append(p)
    df_corr = pd.DataFrame(data={'SID':sids, 'corr':corrs, 'p':ps})
    df_corr.to_csv(f'correlations_{data_type}_{response_tostudy}_{model}.csv', index=False)
    
    for si, sid in enumerate(tqdm(sids)):
        save_path = os.path.join(figure_dir, '%s.png'%sid)
        #if os.path.exists(save_path):
        #    continue
        mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
        spec = mat['spec']
        spec = spec[min(window_start_ids[si]):max(window_start_ids[si])+W]
        spec[np.isinf(spec)] = np.nan
        freq = mat['spec_freq'].flatten()
        
        T = len(Pobs[si])
        tt = np.arange(T)*W*2/3600
        P_ = Pobs[si]*yscale
        import pdb;pdb.set_trace()
        
        plt.close()
        fig = plt.figure(figsize=(14,10))
        gs = GridSpec(4, 2, figure=fig, width_ratios=[4,1])
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(spec.T, cmap='jet', aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax,
                   extent=(tt.min(), tt.max(), freq.min(), freq.max()))
        ax1.set_xlim([tt.min(), tt.max()])
        ax1.set_ylim([freq.min(), freq.max()])
        ax1.set_ylabel('freq (Hz)')
        
        ax2 = fig.add_subplot(gs[1,0])
        #random_ids = np.random.choice(len(Psim[si]), 1, replace=False)
        #ax2.plot(tt, Psim[si][random_ids].T*yscale, c='r', label='simulated (one example)')
        ax2.plot(tt, np.mean(Psim[si], axis=0)*yscale, c='b', ls='--', lw=2, label='mean')# and 95% CI
        ax2.plot(tt, P_, lw=2, c='k', alpha=0.5, label='actual')
        #ax2.plot(tt, np.percentile(Psim[si],2.5,axis=0)*yscale, c='b', ls='--', label='95% CI')
        #ax2.plot(tt, np.percentile(Psim[si],97.5,axis=0)*yscale, c='b', ls='--')
        ax2.fill_between(tt, np.percentile(Psim[si],2.5,axis=0)*yscale, np.percentile(Psim[si],97.5,axis=0)*yscale, color='b', alpha=0.05)
        #TODO
        #if 'lognormal' in model:
        #    tt2 = np.arange(1,T+1)
        #    val = alpha[si] * np.exp(-(np.log(tt2)-mu[si])**2 / (2* sigma[si]**2))#+t0[i]
        #    val = sigmoid(val)
        #    ax2.plot(tt, val*yscale, c='m', label='no drug')
        ax2.legend(fontsize=12, frameon=False, ncol=3)
        #ax2.set_xlabel('time (h)')
        if response_tostudy == 'spike_rate':
            ax2.set_ylabel('Spike rate (/min)')
        elif response_tostudy.startswith('iic_burden'):
            ax2.set_ylabel('IIC burden (%)')
        ax2.set_ylim(ylim)
        ax2.set_xlim([tt.min(), tt.max()])
        
        has_drug_ids = [di for di in range(ND) if np.max(Ddose[si][:,di])>0]
        gs_ax3 = gs[2,0].subgridspec(len(has_drug_ids), 1, hspace=0)
        for axi, di in enumerate(has_drug_ids):
            ax3 = fig.add_subplot(gs_ax3[axi,0])
            dd_mean = Dsim[si][...,di].mean(axis=0)
            dd_lb, dd_ub = np.percentile(Dsim[si][...,di], (2.5,97.5), axis=0)
            ax3.fill_between(tt, dd_lb, dd_ub, color='b', alpha=0.1)
            ax3.plot(tt, dd_mean, c='k')
            ax3.text(0.01, 0.99, Dnames[di], ha='left', va='top', transform=ax3.transAxes, fontsize=12)
            ax3.set_xlim([tt.min(), tt.max()])
            seaborn.despine()
            #ax3.set_yticks([])
            
        gs_ax3_2 = gs[2,1].subgridspec(len(has_drug_ids), 1, hspace=0.5)
        for axi, di in enumerate(has_drug_ids):
            ax3_2 = fig.add_subplot(gs_ax3_2[axi,0])
            ax3_2.hist(simulator.fit_res_df[f'halflife[{si+1},{di+1}]'].values/6, bins=50, color='k', alpha=0.6)
            ax3_2.axvline(halflife_dict[Dname[di]][0]/6, c='r', lw=2, ls='--')
            ax3_2.set_ylabel('count')
            ax3_2.set_xlabel('half-life (hr)')
            seaborn.despine()
        
        """
        ax4 = fig.add_subplot(313)
        for di in range(ND):
            if np.max(Ddose[si][:,di])>0:
                ax4.plot(tt, Ddose[si][:,di]*Dmax[di], label=Dnames[di])
        ax4.legend()#ncol=2
        ax4.set_xlabel('time (h)')
        ax4.set_ylabel('[drug]')
        ax4.set_xlim([tt.min(), tt.max()])
        """
        gs_ax4 = gs[3,0].subgridspec(len(has_drug_ids), 1, hspace=0.5)
        for axi, di in enumerate(has_drug_ids):
            ax4 = fig.add_subplot(gs_ax4[axi,0])
            #ax4.set_title(Dnames[di], fontsize=12)
            #ax4.imshow(Ddose[si][:,di].reshape(1,-1)*Dmax[di], aspect='auto',
            #           extent=(tt.min(), tt.max(), 0,1), cmap='plasma',
            #           vmin=0, vmax=Dmax[di])
            ax4.plot(tt, Ddose[si][:,di], c='k')
            ax4.text(0.01, 0.99, Dnames[di], ha='left', va='top', transform=ax4.transAxes, fontsize=12)
            if axi==len(has_drug_ids)-1:
                ax4.set_xlabel('time (h)')
            else:
                ax4.set_xticklabels([])
            ax4.set_xlim([tt.min(), tt.max()])
            #ax4.set_yticks([])
            seaborn.despine()
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(save_path)
        
