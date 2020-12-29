import os
import pickle
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


#data_type = 'humanIIC'
data_type = 'CNNIIC'
response_tostudy = 'iic_burden'
#response_tostudy = 'spike_rate'

W = 300
max_iter = 1000
Dnames = ['lacosamide', 'levetiracetam', 'midazolam',
            'pentobarbital','phenobarbital',# 'phenytoin',
            'propofol', 'valproate']
ND = len(Dnames)
#models = ['lognormal']#, 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2', 'baseline']
models = ['cauchy_expit_lognormal_drugoutside_ARMA2,6']#'normal_expit_ARMA16', 'student_t_expit_ARMA16', 
cnn_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'
    
with open('../../data_to_fit_CNNIIC_%s.pickle'%response_tostudy, 'rb') as f:
    res = pickle.load(f)
window_start_ids = res['window_start_ids']

for model in models:
    print(model)
    
    figure_dir = 'simulation_global_figures/%s'%model
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    with open(f'../results_{response_tostudy}/results_{data_type}_{model}_iter{max_iter}.pickle', 'rb') as ff:
        res  = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
    
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
    
    for si, sid in enumerate(tqdm(sids)):
        mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
        spec = mat['spec']
        spec = spec[min(window_start_ids[si]):max(window_start_ids[si])+W]
        spec[np.isinf(spec)] = np.nan
        freq = mat['spec_freq'].flatten()
        
        T = len(Pobs[si])
        tt = np.arange(T)*W*2/3600
        P_ = Pobs[si]*100
        
        plt.close()
        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(3, 1, figure=fig)
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(spec.T, cmap='jet', aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax,
                   extent=(tt.min(), tt.max(), freq.min(), freq.max()))
        ax1.set_xlim([tt.min(), tt.max()])
        ax1.set_ylim([freq.min(), freq.max()])
        ax1.set_ylabel('freq (Hz)')
        
        ax2 = fig.add_subplot(gs[1,0])
        random_ids = np.random.choice(len(Psim[si]), 1, replace=False)
        ax2.plot(tt, Psim[si][random_ids].T*100, c='r', label='simulated (one example)')
        ax2.plot(tt, np.mean(Psim[si], axis=0)*100, c='b', ls='--', lw=2, label='mean')# and 95% CI
        ax2.plot(tt, np.percentile(Psim[si],2.5,axis=0)*100, c='b', ls='--', label='95% CI')
        ax2.plot(tt, np.percentile(Psim[si],97.5,axis=0)*100, c='b', ls='--')
        ax2.plot(tt, P_, lw=2, c='k', alpha=0.5, label='actual')
        #TODO
        #if 'lognormal' in model:
        #    tt2 = np.arange(1,T+1)
        #    val = alpha[si] * np.exp(-(np.log(tt2)-mu[si])**2 / (2* sigma[si]**2))#+t0[i]
        #    val = sigmoid(val)
        #    ax2.plot(tt, val*100, c='m', label='no drug')
        ax2.legend(fontsize=12, frameon=False, ncol=3)
        #ax2.set_xlabel('time (h)')
        ax2.set_ylabel('IIC burden (%)')
        ax2.set_ylim([-2,102])
        ax2.set_xlim([tt.min(), tt.max()])
        
        """
        ax3 = fig.add_subplot(313)
        for di in range(ND):
            if np.max(Dscaled[si][:,di])>0:
                ax3.plot(tt, Dscaled[si][:,di]*Dmax[di], label=Dnames[di])
        ax3.legend()#ncol=2
        ax3.set_xlabel('time (h)')
        ax3.set_ylabel('[drug]')
        ax3.set_xlim([tt.min(), tt.max()])
        """
        has_drug_ids = [di for di in range(ND) if np.max(Dscaled[si][:,di])>0]
        gs_ax3 = gs[2,0].subgridspec(len(has_drug_ids), 1, hspace=0.5)
        for axi, di in enumerate(has_drug_ids):
            ax3 = fig.add_subplot(gs_ax3[axi,0])
            ax3.set_title(Dnames[di], fontsize=12)
            ax3.imshow(Dscaled[si][:,di].reshape(1,-1)*Dmax[di], aspect='auto',
                       extent=(tt.min(), tt.max(), 0,1), cmap='plasma',
                       vmin=0, vmax=Dmax[di])
            if axi==len(has_drug_ids)-1:
                ax3.set_xlabel('time (h)')
            else:
                ax3.set_xticklabels([])
            ax3.set_xlim([tt.min(), tt.max()])
            ax3.set_yticks([])
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_dir, '%s.png'%sids[si]))
        
