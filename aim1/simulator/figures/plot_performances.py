from itertools import combinations
import os
import pickle
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


models = ['lognormal', 'AR1', 'AR2', 'PAR1', 'PAR2', 'lognormalAR1','lognormalAR2']#, 'baseline']
metrics = ['loglikelihood']
metric_names = ['log-likelihood']

with open('../performance_metrics.pickle', 'rb') as ff:
    perf = pickle.load(ff)
sids = perf['sids']
N = len(sids)

"""
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
"""
        
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
    
