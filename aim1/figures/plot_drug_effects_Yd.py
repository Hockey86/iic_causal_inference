import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


outcome_model_type = str(sys.argv[1])
AR_p = 2
MA_q = 6
simulator_model_type = f'cauchy_expit_lognormal_drugoutside_ARMA{AR_p},{MA_q}'
data_type = 'CNNIIC'
max_iter = 1000
#drugname = 'levetiracetam'
drugname ='propofol'

if drugname == 'levetiracetam':
    drug_regimes = ['always_zero', 'keppra_8h_30', 'keppra_8h_60', 'keppra_8h_90']
    concentrations = [0,30,60,90]
    xlabel = 'Treatment regime:\nGive keppra (mg/kg)\nevery 8h'
elif drugname == 'propofol':
    drug_regimes = ['always_zero', 'always_propofol_1', 'always_propofol_5', 'always_propofol_10']
    concentrations = [0,1,5,10]
    xlabel = 'Treatment regime:\nConstant propofol\nconcentration'

with open(f'../res_evaluate_Yd_{outcome_model_type}_{simulator_model_type}.pickle', 'rb') as ff:
    Yd, Ds, Es, Xs = pickle.load(ff)
    
df_sim_iic = pd.read_csv(f'../step6_simulator/results_iic_burden_smooth/params_mean_{data_type}_iic_burden_smooth_{simulator_model_type}_iter{max_iter}.csv')[:500]
df_sim_spikerate = pd.read_csv(f'../step6_simulator/results_spike_rate/params_mean_{data_type}_spike_rate_{simulator_model_type}_iter{max_iter}.csv')[:500]
    
# limit to patients who actually received the drug
ids = np.where(~pd.isna(df_sim_iic[f'b[{drugname}]']))[0]
nodrug_iic_burden = Xs['always_zero'].mean(axis=1)[ids,7]
nodrug_spr_burden = Xs['always_zero'].mean(axis=1)[ids,9]
drug_iic_coef = df_sim_iic[f'b[{drugname}]'].values[ids]
drug_spr_coef = df_sim_spikerate[f'b[{drugname}]'].values[ids]

ids = ids[(drug_iic_coef>1)&(nodrug_iic_burden>0.1)&(drug_spr_coef>1)&(nodrug_spr_burden>0.1)]
Yd = {ln:Yd[ln][ids] for ln in drug_regimes}
Xs = {ln:Xs[ln][ids] for ln in drug_regimes}
 
iics = []
spike_rates = []
iic_spike_rates = []
Yds = []
for dr in drug_regimes:
    iics.append(Xs[dr][...,7].mean(axis=0))
    spike_rates.append(Xs[dr][...,9].mean(axis=0))
    iic_spike_rates.append(Xs[dr][...,11].mean(axis=0))
    Yds.append(Yd[dr].mean(axis=0))

ylim = (0.2,0.7)
plt.close()
fig = plt.figure(figsize=(13,6))

ax = fig.add_subplot(141)
ax.boxplot(iics)
ax.set_xticklabels([str(x) for x in concentrations])
ax.set_xlabel(xlabel)
ax.set_ylabel('Average IIC burden\nacross subjects')
#ax.set_ylim(0.5,0.75)
seaborn.despine()

ax = fig.add_subplot(142)
ax.boxplot(spike_rates)
ax.set_xticklabels([str(x) for x in concentrations])
ax.set_xlabel(xlabel)
ax.set_ylabel('Average spike rate burden (/s)\nacross subjects')
#ax.set_ylim(0.5,0.75)
seaborn.despine()

ax = fig.add_subplot(143)
ax.boxplot(iic_spike_rates)
ax.set_xticklabels([str(x) for x in concentrations])
ax.set_xlabel(xlabel)
ax.set_ylabel('Average IIC x spike rate burden\nacross subjects')
#ax.set_ylim(0.25,0.5)
seaborn.despine()

ax = fig.add_subplot(144)
ax.boxplot(Yds)
ax.set_xticklabels([str(x) for x in concentrations])
ax.set_xlabel(xlabel)
ax.set_ylabel('P(Potential discharge mRS>=4)')
#ax.set_ylim(0.8,0.9)
#ax.set_yticks([0.7,0.75,0.8,0.85,0.9])
seaborn.despine()

plt.tight_layout()
#plt.show()
plt.savefig(f'Yd_{drugname}2.png', bbox_inches='tight', pad_inches=0.05)
