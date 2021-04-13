import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
from tqdm import tqdm


if __name__=='__main__':
    data_type = 'CNNIIC'
    responses = ['iic_burden_smooth', 'spike_rate']
    responses_txt = '+'.join(responses)
    AR_p = 2
    MA_q = 6
    simulator_model_type = f'cauchy_expit_lognormal_drugoutside_ARMA{AR_p},{MA_q}'
    max_iter = 1000
    Nbt = 0
    n_jobs = 12
    random_state = 2020
    
    ## load data
    
    df_sim_iic = pd.read_csv(f'../step6_simulator/results_iic_burden_smooth/params_mean_{data_type}_iic_burden_smooth_{simulator_model_type}_iter{max_iter}.csv')[:500]
    df_sim_spikerate = pd.read_csv(f'../step6_simulator/results_spike_rate/params_mean_{data_type}_spike_rate_{simulator_model_type}_iter{max_iter}.csv')[:500]
    with open(f'../res_evaluate_Yd_rf_{simulator_model_type}.pickle', 'rb') as ff:
        Yd, Ds, Es, Xs = pickle.load(ff)
        
    cmap = matplotlib.cm.get_cmap('rainbow')
    
    for drugname in ['levetiracetam', 'propofol']:
        figure_dir = f'simulated_IIC_{drugname}'
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
                
        if drugname=='levetiracetam':
            drugid = 1
            drug_regimes = ['always_zero', 'keppra_8h_30', 'keppra_8h_60', 'keppra_8h_90']
            doses = [0,30,60,90]
            xlabel = 'keppra dose (mg/kg) given every 8h'
        elif drugname=='propofol':
            drugid = 5
            drug_regimes = ['always_zero', 'always_propofol_1', 'always_propofol_5', 'always_propofol_10']
            doses = [0,1,5,10]
            xlabel = 'propofol constant concentration'
        
        
        plt.close()
        fig = plt.figure(figsize=(9,9))
        
        for bi, burden_type in enumerate(['iic_ratio_mean', 'iic_ratio_max', 'spike_rate_mean', 'spike_rate_max']):
            ax = fig.add_subplot(2,2,bi+1)
            if burden_type=='iic_ratio_mean':
                burden_id = 7
                ylabel = 'IIC burden (mean)'
                ylim = [0,1]
                scale = 1
            elif burden_type=='iic_ratio_max':
                burden_id = 8
                ylabel = 'IIC burden (hourly max)'
                ylim = [0,1]
                scale = 1
            elif burden_type=='spike_rate_mean':
                burden_id = 9
                ylabel = 'spike rate burden (mean)'
                ylim = [0,60]
                scale = 60
            elif burden_type=='spike_rate_max':
                burden_id = 10
                ylabel = 'spike rate burden (hourly max)'
                ylim = [0,60]
                scale = 60
            
            ids = np.where(~pd.isna(df_sim_iic[f'b[{drugname}]']))[0]
            iic_burdens = [Xs[ln].mean(axis=1)[ids,burden_id] for ln in drug_regimes]
            drug_coef = df_sim_iic[f'b[{drugname}]'].values[ids]
            
            for i in range(len(iic_burdens[0])):
                alpha = drug_coef[i]/100*0.95+0.05
                ax.plot(doses, [x[i]*scale for x in iic_burdens], c='k', alpha=alpha)
                ax.scatter(doses, [x[i]*scale for x in iic_burdens], s=15, c='k', alpha=alpha)
            ax.set_xlim(doses[0]-(doses[-1]-doses[0])*0.05, doses[-1]+(doses[-1]-doses[0])*0.05)
            ax.set_xticks(doses)
            ax.set_xticklabels(doses)
            ax.set_xlabel(xlabel)
            ax.set_ylim(ylim)
            ax.set_ylabel(ylabel)
            seaborn.despine()
            
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_dir, f'IIC_{drugname}.png'), bbox_inches='tight', pad_inches=0.05)
     
        continue
        np.random.seed(random_state)
        Nsamp = 10
        N = len(Ds['always_zero'])
        for i in tqdm(range(N)):
            if pd.isna(df_sim_iic[f'b[{drugname}]'].iloc[i]):
                continue
            sid = df_sim_iic.SID.iloc[i]
            T = len(Ds['always_zero'][i])
            tt = np.arange(T)*600/3600
            
            plt.close()
            fig = plt.figure(figsize=(12,8))
            
            # plot simulated IIC
            ax = fig.add_subplot(311)
            for li, ln in enumerate(drug_regimes):
                iic = Es[ln][i]['iic_burden_smooth']
                iic_mean = iic.mean(axis=0)
                iic_samp = iic[np.random.choice(len(iic), Nsamp, replace=False)].T
                
                color = cmap(li/(len(drug_regimes)-1))
                #ax.plot(tt, iic_samp, c=color, alpha=0.2)
                ax.plot(tt, iic_mean, c=color, lw=2, label=ln)
            ax.set_xlim(tt.min(), tt.max())
            ax.set_ylim(0,1)
            #ax.legend(frameon=False)
            ax.set_ylabel('IIC burden')
            seaborn.despine()
            
            # plot simulated spike rate
            ax = fig.add_subplot(312)
            for li, ln in enumerate(drug_regimes):
                iic = Es[ln][i]['spike_rate']*60
                iic_mean = iic.mean(axis=0)
                iic_samp = iic[np.random.choice(len(iic), Nsamp, replace=False)].T
                
                color = cmap(li/(len(drug_regimes)-1))
                #ax.plot(tt, iic_samp, c=color, alpha=0.2)
                ax.plot(tt, iic_mean, c=color, lw=2, label=ln)
            ax.set_xlim(tt.min(), tt.max())
            ax.set_ylim(0,60)
            #ax.legend(frameon=False)
            ax.set_ylabel('spike rate (/min)')
            seaborn.despine()
            
            # plot input drug
            ax = fig.add_subplot(313)
            for li, ln in enumerate(drug_regimes):
                drug = Ds[ln][i][:,drugid]
                color = cmap(li/(len(drug_regimes)-1))
                ax.plot(tt, drug, c=color, lw=2, label=ln)
            ax.set_xlim(tt.min(), tt.max())
            ax.set_xlabel('time (h)')
            ax.legend(frameon=False)
            ax.set_ylabel(f'{drugname} (a.u.)')
            seaborn.despine()
            
            plt.tight_layout()
            #plt.show()
            plt.savefig(os.path.join(figure_dir, f'{sid}.png'), bbox_inches='tight', pad_inches=0.05)
    
