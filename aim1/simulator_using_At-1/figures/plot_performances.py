
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


models = ['normal_expit_ARMA16','baseline']
models_txt = ['normal expit ARMA(1,6)','baseline(2)']
markers = ['o', '^']
colors = ['r', 'k']

## load

metrics = [ 'KL-divergence', 'loglikelihood', 'CI95 Coverage', 'stRMSE']
with open("../results/performance_metrics_%s.pickle"%models, 'rb') as ff:
    perf = pickle.load(ff)
sids = perf['sids']
Ts = perf['Ts_stRMSE']*10 # each time step is 10 minutes

## print CIs

for model in models:
    for metric in metrics:
        if metric in ['loglikelihood', 'CI95 Coverage', 'KL-divergence', 'spearmanr']:
            if perf[(model, metric)].ndim>1:
                perf_ = np.nanmean(perf[(model, metric)], axis=0)
                print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))
            else:
                print('[%s, %s]: %.2f [%.2f -- %.2f]'%(model, metric, perf[(model, metric)].mean(), np.nanpercentile(perf[(model, metric)], 2.5), np.nanpercentile(perf[(model, metric)], 97.5)))


        elif metric == 'WAIC':
            print('[%s, %s]: %.2f'%(model, metric, perf[(model, metric)]))

        elif metric == 'stRMSE':
            for ii, t in enumerate(Ts):
                perf_ = np.nanmean(perf[(model, metric)][ii], axis=0)
                print('[%s, %s(%d)]: %.2f [%.2f -- %.2f]'%(model, metric, t, perf_.mean(), np.nanpercentile(perf_, 2.5), np.nanpercentile(perf_, 97.5)))


## average across patients stRMSE

plt.close()
plt.figure(figsize=(8,6))
for mi, model in enumerate(models):
    mean_ = [np.mean(perf[(model, 'stRMSE')][ti].mean(axis=0)) for ti in np.arange(len(Ts))]
    ub_ = [np.percentile(perf[(model, 'stRMSE')][ti].mean(axis=0), 97.5) for ti in np.arange(len(Ts))]
    lb_ = [np.percentile(perf[(model, 'stRMSE')][ti].mean(axis=0), 2.5) for ti in np.arange(len(Ts))]

    plt.plot(Ts,mean_,marker=markers[mi],c=colors[mi],label=models_txt[mi])
    plt.plot(Ts,ub_,c=colors[mi])
    plt.plot(Ts,lb_,c=colors[mi])

plt.legend()
plt.xlabel('Time (min)')
plt.ylabel('stRMSE')
plt.ylim([0,1])
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig('stRMSE_average.png')


## per patient stRMSE
folder = 'stRMSE_per_patient'
if not os.path.exists(folder):
    os.mkdir(folder)
for si, sid in enumerate(tqdm(sids)):
    plt.close()
    plt.figure(figsize=(8,6))
    for mi, model in enumerate(models):
        print(model)
        mean_ = [np.mean(perf[(model, 'stRMSE')][ti][si]) for ti in np.arange(len(Ts))]
        print(mean_)
        ub_ = [np.percentile(perf[(model, 'stRMSE')][ti][si], 97.5) for ti in np.arange(len(Ts))]
        lb_ = [np.percentile(perf[(model, 'stRMSE')][ti][si], 2.5) for ti in np.arange(len(Ts))]

        plt.plot(Ts,mean_,marker=markers[mi],c=colors[mi],label=models_txt[mi])
        plt.plot(Ts,ub_,c=colors[mi])
        plt.plot(Ts,lb_,c=colors[mi])

    plt.legend()
    plt.xlabel('Time (min)')
    plt.ylabel('stRMSE')
    plt.ylim([0,1])
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(folder, 'stRMSE_%s.png'%sid))
