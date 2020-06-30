from itertools import groupby
import datetime
from glob import glob
import os
import numpy as np
import pandas as pd
from scipy.special import logit
import scipy.io as sio
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')
from tqdm import tqdm


# define input files
data_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
sids = ['sid36', 'sid39', 'sid56', 'sid297', 'sid327', 'sid385',
    'sid395', 'sid400', 'sid403', 'sid406', 'sid424', 'sid450',
    'sid456', 'sid490', 'sid512', 'sid551', 'sid557', 'sid575',
    'sid988', 'sid1016', 'sid1025', 'sid1034', 'sid1038', 'sid1039',
    'sid1055', 'sid1056', 'sid1063', 'sid1337', 'sid1897', 'sid1913',
    'sid1915', 'sid1916', 'sid1917', 'sid1926', 'sid1928', 'sid1956',
    'sid1966']+\
   ['sid2', 'sid23', 'sid45', 'sid77', 'sid91', 'sid741', 'sid821', 'sid832', 'sid848',
    'sid8', 'sid24', 'sid54', 'sid82', 'sid92', 'sid771', 'sid822', 'sid833', 'sid849',
    'sid11', 'sid28', 'sid57', 'sid84', 'sid97', 'sid801', 'sid823', 'sid834', 'sid852',
    'sid13', 'sid30', 'sid61', 'sid88', 'sid734', 'sid808', 'sid824', 'sid837', 'sid856',
    'sid17', 'sid38', 'sid69', 'sid89', 'sid736', 'sid815', 'sid827', 'sid839',
    'sid18', 'sid44', 'sid71', 'sid90', 'sid739', 'sid817', 'sid828', 'sid845']+\
    ['sid863', 'sid864', 'sid865', 'sid870', 'sid872', 'sid875', 'sid876', 'sid880',
     'sid881', 'sid884', 'sid886', 'sid887', 'sid890', 'sid914', 'sid915', 'sid917',
     'sid918', 'sid927', 'sid933', 'sid940', 'sid942', 'sid944', 'sid952', 'sid960',
     'sid963', 'sid965', 'sid967', 'sid983', 'sid984', 'sid987', 'sid994', 'sid1000',
     'sid1002', 'sid1006', 'sid1022', 'sid1024', 'sid1101', 'sid1102', 'sid1105',
     'sid1113', 'sid1116']


# define windows
# if using the first 24h, we cannot have window size of 12h or 24h
window_times = [1*60, 10*60, 3600, 3*3600, 6*3600]#, 12*3600, 24*3600  # [s]
num_windows = [10,6,4,4,2]#,2,2
window_txt = ['1min', '10min', '1h', '3h', '6h']#, '12h', '24h'

# color scale for spectrogram
vmin = -10
vmax = 25

# create custom color map
N = 6
vals = np.array([
    [0,0.5,1,1],
    [1,0,0,1],
    [1,0.5,0,1],
    [1,1,0,1],
    [0.5,1,0.5,1],
    [0,1,1,1],
])
iic_cm = ListedColormap(vals)

tostudy_Dnames = [
            'levetiracetam', 'lacosamide',
            'fosphenytoin',# 'phenytoin',# these are the same
            'valproate',# 'divalproex',# these are the same
            'propofol', 'midazolam']#, 'pentobarbital']


# for each patient
for sid in tqdm(sids):
    res = sio.loadmat(os.path.join(data_dir, sid+'.mat'))
    spec = res['spec']
    spec[np.isinf(spec)] = np.nan
    freq = res['spec_freq'].flatten()
    spike = res['spike'].flatten()
    start_time = res['start_time'][0].strip()
    Dnames = [x.strip() for x in res['Dnames']]
    human_label = res['human_iic'].flatten().astype(float)
    artifact_indicator = res['artifact'].flatten()
    
    human_label[artifact_indicator==1] = np.nan
    spike[artifact_indicator==1] = np.nan
    
    res_sim = sio.loadmat(os.path.join('/data/IIC-Causality/mycode/aim1/simulator/simluations', 'E_%s.mat'%sid))
    iic_sim = res_sim['Esim'].flatten()
    iic_sim = np.repeat(iic_sim, 900)[:len(human_label)]
    
    
    # for each drug
    for dn in tostudy_Dnames:
        di = Dnames.index(dn)
        this_drug = res['drug'][:,di]
        
        # get drug dose change points
        #TODO use increase only
        change_ids = np.where(np.diff(this_drug)>0)[0]#|(np.diff(this_drug<0))
        
        # for each change point
        for ci in change_ids:
            current_time = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')+datetime.timedelta(seconds=int(ci)*2)
            current_time_txt = datetime.datetime.strftime(current_time, '%Y-%m-%d %H-%M-%S')
        
            # define output path
            save_path = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/drug_change_points/%s/drug_changepoint_%s_%s.png'%(dn, sid, current_time_txt)
            # if output path exists, go to next loop
            #if os.path.exists(save_path):
            #    continue
            window_time = window_times[-1]
            n_w = num_windows[-1]
            total_window_size = int(window_time*n_w)//2
            window_size = int(window_time)//2
            
            start = max(0, ci-total_window_size)
            end = min(len(this_drug), ci+total_window_size)
            
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
            drug_ = res['drug'][start:end]
            
            # for each time scale
            
            start_ids = []
            sz_burdens = []
            iic_burdens = []
            iic_sim_burdens = []
            spike_rates = []
            for wi in range(len(window_times)):
                window_size_ = int(window_times[wi])//2
                # start_id is the array of start ids of each window
                start_id = np.r_[ci-np.arange(0, ci-start-window_size_+1, window_size_)[::-1]-window_size_,
                                  ci+np.arange(0, end-ci-window_size_+1, window_size_)]
                start_ids.append(start_id)             
                
                # segment human_label into windows
                # human_label.shape = (#2s-window,)
                # human_label.reshape(1,-1).shape = (1, #2s-window)
                # human_label.reshape(1,-1)[...].shape = (1, #window, window_size), note window_size=#2s-window in a window
                # human_label.reshape(1,-1)[...][0].shape = (#window, window_size)
                human_label_segs = human_label.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+window_size_), start_id))][0]
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
                
                iic_sim_segs = iic_sim.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+window_size_), start_id))][0]
                # then take the mean, mean of binary array = % of 1's
                iic_sim_burden = np.nanmean(iic_sim_segs, axis=1)*100
                iic_sim_burdens.append(iic_sim_burden)
                
                # segment spike_rate into windows
                # spike_rate.shape = (#2s-window,)
                spike_rate = spike.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+window_size_), start_id))][0]
                # spike_rate.shape = (#window, window_size)
                nanids = np.all(np.isnan(spike_rate), axis=1)
                spike_rate = np.nansum(spike_rate, axis=1)/window_size_*60
                spike_rate[nanids] = np.nan
                spike_rates.append(spike_rate)
            
            # generate the figure
            
            plt.close()
            fig = plt.figure(figsize=(13,8))
            gs = fig.add_gridspec(ncols=1, nrows=5, height_ratios=[3,5,5,1,3])#,1
            
            xticks = start_ids[2]
            xticklabels = np.array([str(x) for x in (xticks-ci)*2], dtype=object)
            xticklabels[xticklabels=='0'] = current_time_txt.replace(' ','\n')
            
            # ax1 is spectrogram
            
            ax1 = fig.add_subplot(gs[0])
            ax1.imshow(spec_.T, aspect='auto', origin='lower', cmap='jet',
                       vmin=vmin, vmax=vmax,
                       extent=(start, end, freq.min(), freq.max()))
            ax1.set_ylabel('Hz')
            
            # plot artifact indicator
            cc = 0
            for k,l in groupby(artifact):
                ll = len(list(l))
                if k==1:
                    ax1.plot([cc+start, cc+ll+start], [freq.max()+1.5]*2, lw=5, c='k')
                cc += ll
                
            ax1.set_xticks(xticks)
            ax1.set_xticklabels([])
            #for st in start_id:
            #    ax1.axvline(st, ls='--', color='r' if st==ci else 'k', lw=1 if st==ci else 0.5)
            ax1.axvline(ci, ls='--', color='r' , lw=1)
            #ax1.set_xlim([start_id.min(), start_id.max()+window_size])
            ax1.set_xlim([start, end])
            ax1.set_ylim([freq.min(), freq.max()+3])
            ax1.set_title('SID: %s    Change Time: %s    Drug: %s'%(sid, current_time_txt, Dnames[di]))
            
            # ax2 is Sz burden based on human label
            subgs = gs[1].subgridspec(len(window_times), 1, hspace=0)
            for wi in range(len(window_times)):
                start_id = start_ids[wi]
                window_size_ = int(window_times[wi])//2
                sz_burden = sz_burdens[wi]
                iic_burden = iic_burdens[wi]
                iic_sim_burden = iic_sim_burdens[wi]
                ax2 = fig.add_subplot(subgs[wi])
                if wi>=2:
                    marker = 'o'
                else:
                    marker = ''
                ms = 5
                ax2.plot(start_id+window_size_//2, sz_burden, marker=marker, ms=ms, c='r', label='Sz')
                ax2.plot(start_id+window_size_//2, iic_burden, marker=marker, ms=ms, c='k', label='IIC')
                ax2.plot(start_id+window_size_//2, iic_sim_burden, marker=marker, ms=ms, c='b', label='SimIIC')
                ax2.text(0, 1, window_txt[wi], ha='left', va='top', transform=ax2.transAxes)
                if wi==len(window_times)-1:
                    ax2.legend(ncol=2)
                if wi==len(window_times)//2:
                    ax2.set_ylabel('Burden (%)')
                else:
                    ax2.set_ylabel('')
                ax2.set_xticks(xticks)
                ax2.set_xticklabels([])
                #for st in start_id:
                #    ax2.axvline(st, ls='--', color='r' if st==ci else 'k')
                ax2.axvline(ci, ls='--', color='r')
                #ax2.set_xlim([start_ids[-1].min(), start_ids[-1].max()+window_size])
                ax2.set_xlim([start, end])
                # if constant, set ylim to constant +/- 1
                if np.nanmin(sz_burden)==np.nanmax(sz_burden)==np.nanmin(iic_burden)==np.nanmax(iic_burden):
                    ax2.set_ylim([np.nanmin(sz_burden)-1, np.nanmin(sz_burden)+1])
                else:
                    ax2.set_ylim([-5,105])
                #seaborn.despine()
            
            # ax3 is spike rate
            subgs = gs[2].subgridspec(len(window_times), 1, hspace=0)
            for wi in range(len(window_times)):
                start_id = start_ids[wi]
                window_size_ = int(window_times[wi])//2
                spike_rate = spike_rates[wi]
                ax3 = fig.add_subplot(subgs[wi])
                if wi>=2:
                    marker = 'o'
                else:
                    marker = ''
                ms = 5
                ax3.plot(start_id+window_size_//2, spike_rate, marker=marker, ms=ms)
                ax3.text(0, 1, window_txt[wi], ha='left', va='top', transform=ax3.transAxes)
                if wi==len(window_times)//2:
                    ax3.set_ylabel('Spike Rate\n(/min)')
                else:
                    ax3.set_ylabel('')
                ax3.set_xticks(xticks)
                ax3.set_xticklabels([])
                #for st in start_id:
                #    ax3.axvline(st, ls='--', color='r' if st==ci else 'k')
                ax3.axvline(ci, ls='--', color='r')
                #ax3.set_xlim([start_ids[-1].min(), start_ids[-1].max()+window_size])
                ax3.set_xlim([start, end])
                # if constant, set ylim to constant +/- 1
                if np.nanmin(spike_rate)==np.nanmax(spike_rate):
                    ax3.set_ylim([np.nanmin(spike_rate)-1, np.nanmin(spike_rate)+1])
                else:
                    ax3.set_ylim([-2,62])
                ax3.set_xticklabels([])
                #seaborn.despine()

            # ax4 is human label
            ax4 = fig.add_subplot(gs[3])
            ax4.imshow(human_label_.reshape(1,-1), aspect='auto', cmap=iic_cm,
                       vmin=0, vmax=5,
                       extent=(start, end, 0,1))
            #plt.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=5), cmap=iic_cm),
            #             orientation="horizontal", ax=ax4, anchor=(0,1) )
            ax4.set_ylim([0,1])
            ax4.set_ylabel('Human')#, rotation=0)
            #ax4.set_xlim([start_ids[-1].min(), start_ids[-1].max()+window_size])
            ax4.set_xlim([start, end])
            ax4.set_xticks(xticks)
            ax4.set_xticklabels([])
            
            """
            # ax5 is CNN prediction
            ax5 = fig.add_subplot(gs[4])
            ax5.imshow(iic_cnn_prediction.reshape(1,-1), aspect='auto', cmap=iic_cm,
                       vmin=0, vmax=5,
                       extent=(start, end, 0,1))
            ax5.set_ylim([0,1])
            ax5.set_ylabel('CNN')#, rotation=0)
            #ax5.set_xlim([start_ids[-1].min(), start_ids[-1].max()+window_size])
            ax5.set_xlim([start, end])
            ax5.set_xticks(xticks)
            ax5.set_xticklabels([])
            """
            
            # ax6 is drug
            drug_notzero_ids = np.where(np.nansum(drug_, axis=0)>0)[0]
            subgs = gs[4].subgridspec(len(drug_notzero_ids), 1, hspace=0)
            for j in range(len(drug_notzero_ids)):
                ax6 = fig.add_subplot(subgs[j])
                ax6.plot(np.arange(start, end), drug_[:,drug_notzero_ids[j]])
                ax6.set_ylabel(Dnames[drug_notzero_ids[j]], rotation=0, ha='right')
                ax6.set_xticks(xticks)
                if j==len(drug_notzero_ids)-1:
                    ax6.set_xticklabels(xticklabels)
                else:
                    ax6.set_xticklabels([])
                #ax6.set_xlim([start_ids[-1].min(), start_ids[-1].max()+window_size])
                ax6.set_xlim([start, end])
                            
            # if the drug folder does not exist, create it
            dir_name = os.path.dirname(save_path)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            # save figure
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.23)
            #plt.show()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
            
