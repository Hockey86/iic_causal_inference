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
simulator_model_type = 'cauchy_expit_lognormal_drugoutside_ARMA'
drug_regimes = ['always_zero', 'always_propofol_1', 'always_propofol_2', 'always_propofol_3']

with open(f'res_evaluate_Yd_{outcome_model_type}_{simulator_model_type}.pickle', 'rb') as ff:
    Yd, Ds, Es, Xs = pickle.load(ff)
 
concentrations = []
iics = []
spike_rates = []
iic_spike_rates = []
Yds = []
import pdb;pdb.set_trace()
for dr in drug_regimes:
    concentrations.append(Ds[dr][0][1,5])
    iics.append(Xs[dr][...,7].mean(axis=0))
    spike_rates.append(Xs[dr][...,8].mean(axis=0))
    iic_spike_rates.append(Xs[dr][...,9].mean(axis=0))
    Yds.append(Yd[dr].mean(axis=0))
#iics = np.percentile(np.mean(iics, axis=1),(2.5,50,97.5),axis=1).T

ylim = (0.2,0.7)
plt.close()
fig = plt.figure(figsize=(13,6))

ax = fig.add_subplot(141)
ax.boxplot(iics, positions=concentrations)
ax.set_xlabel('Treatment regime:\nConstant propofol\nbrain concentration (mg/kg)')
ax.set_ylabel('Average IIC burden')
ax.set_ylim(0.45,0.7)
seaborn.despine()

ax = fig.add_subplot(142)
ax.boxplot(spike_rates, positions=concentrations)
ax.set_xlabel('Treatment regime:\nConstant propofol\nbrain concentration (mg/kg)')
ax.set_ylabel('Average spike rate burden (/min)')
ax.set_ylim(0.45,0.7)
seaborn.despine()

ax = fig.add_subplot(143)
ax.boxplot(iic_spike_rates, positions=concentrations)
ax.set_xlabel('Treatment regime:\nConstant propofol\nbrain concentration (mg/kg)')
ax.set_ylabel('Average IIC x spike rate burden')
ax.set_ylim(0.2,0.45)
seaborn.despine()

ax = fig.add_subplot(144)
ax.boxplot(Yds, positions=concentrations)
ax.set_xlabel('Treatment regime:\nConstant propofol\nbrain concentration (mg/kg)')
ax.set_ylabel('P(Potential discharge mRS>=4)')
ax.set_ylim(0.7,0.9)
ax.set_yticks([0.7,0.75,0.8,0.85,0.9])
seaborn.despine()

plt.tight_layout()
plt.show()
