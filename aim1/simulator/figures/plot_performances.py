from itertools import combinations
import os
import pickle
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


with open('../results/performance_metrics_separteAR1MA6_AR2MA6_baseline.pickle', 'rb') as ff:
    perf = pickle.load(ff)

tt_ar1 = np.arange(1,13)
mean_ar1 = [np.mean(perf[('AR1', 'stRMSE(%d)'%t)].mean(axis=0)) for t in tt_ar1]
ub_ar1 = [np.percentile(perf[('AR1', 'stRMSE(%d)'%t)].mean(axis=0), 97.5) for t in tt_ar1]
lb_ar1 = [np.percentile(perf[('AR1', 'stRMSE(%d)'%t)].mean(axis=0), 2.5) for t in tt_ar1]

tt_ar2 = np.arange(1,13)
mean_ar2 = [np.mean(perf[('AR2', 'stRMSE(%d)'%t)].mean(axis=0)) for t in tt_ar2]
ub_ar2 = [np.percentile(perf[('AR2', 'stRMSE(%d)'%t)].mean(axis=0), 97.5) for t in tt_ar2]
lb_ar2 = [np.percentile(perf[('AR2', 'stRMSE(%d)'%t)].mean(axis=0), 2.5) for t in tt_ar2]

tt_bl = np.arange(2,13)
mean_bl = [np.mean(perf[('baseline', 'stRMSE(%d)'%t)].mean(axis=0)) for t in tt_bl]
ub_bl = [np.percentile(perf[('baseline', 'stRMSE(%d)'%t)].mean(axis=0), 97.5) for t in tt_bl]
lb_bl = [np.percentile(perf[('baseline', 'stRMSE(%d)'%t)].mean(axis=0), 2.5) for t in tt_bl]

plt.close()
plt.figure(figsize=(8,6))
plt.plot(tt_ar1,mean_ar1,marker='o',c='r',label='AR(1) + MA(6)')
plt.plot(tt_ar2,mean_ar2,marker='*',c='b',label='AR(2) + MA(6)')
plt.plot(tt_bl,mean_bl,marker='^',c='k',label='Baseline(2)')
#plt.plot(tt,lb,c='k');plt.plot(tt,ub,c='k')
plt.legend()
plt.xlabel('step, each step is 10min')
plt.ylabel('stRMSE')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig('stRMSE.png')

"""
models = ['lognormal', 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2']#, 'baseline']
metrics = ['loglikelihood']
metric_names = ['log-likelihood']

with open('../performance_metrics.pickle', 'rb') as ff:
    perf = pickle.load(ff)
sids = perf['sids']
N = len(sids)

# perf per patient
output_dir = 'performance_plots_per_patient'
for mi, metric in enumerate(metrics):
    print(metric)
    output_dir_ = os.path.join(output_dir, metric)
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)
        
    for i in tqdm(range(N)):
        data = [perf[(m, metric)][i] for m in models]
        
        plt.close()
        fig = plt.figure(figsize=(9,6))
        
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=models, showfliers=False)
        #ax.set_xlabel('')
        ax.set_ylabel(metric_names[mi])
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(output_dir_, '%s.png'%sids[i]))
        
# perf across patients
for mi, metric in enumerate(metrics):
    print(metric)
    
    data = [perf[(m, metric)].mean(axis=1) for m in models]
    pvals = {(i1,i2): wilcoxon(data[i1], data[i2]).pvalue for i1, i2 in combinations(range(len(models)), 2)}
    pvals = {k:pvals[k]*len(pvals) for k in pvals}
    
    for i1, i2 in combinations(range(len(models)), 2):
        print('%s vs %s: p = %f, %s'%(models[i1], models[i2], pvals[(i1,i2)], '*' if pvals[(i1,i2)]<0.05 else ''))
    
    plt.close()
    fig = plt.figure(figsize=(9,6))
    
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=models, showfliers=False)
    #ax.set_xlabel('')
    ax.set_ylabel(metric_names[mi])
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('performance_across_patients.png')
"""
