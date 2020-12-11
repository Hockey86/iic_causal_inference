import pickle
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn
seaborn.set_style('ticks')

with open('../results_iic_burden/model_fit_CNNIIC_cauchy_expit_lognormal_drugoutside_ARMA2,6_iter1000.pkl','rb') as ff:
    _, df = pickle.load(ff)

"""
['sigma_alpha0[1]', 'sigma_alpha0[2]']
['sigma_alpha[1]', 'sigma_alpha[2]']
['sigma_b[1,1]', 'sigma_b[1,2]', 'sigma_b[1,3]', 'sigma_b[1,4]', 'sigma_b[1,5]', 'sigma_b[1,6]', 'sigma_b[1,7]', 'sigma_b[2,1]', 'sigma_b[2,2]', 'sigma_b[2,3]', 'sigma_b[2,4]', 'sigma_b[2,5]', 'sigma_b[2,6]', 'sigma_b[2,7]']
['sigma_err[1]', 'sigma_err[2]']
"""

drugnames = ['lacosamide', 'levetiracetam', 'midazolam',
             'pentobarbital','phenobarbital',# 'phenytoin',
             'propofol', 'valproate']
                
plt.close()
figsize = (8,6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.boxplot([df['sigma_alpha0[1]'],df['sigma_alpha0[2]']], labels=['Cluster 1','Cluster 2'])
pval = mannwhitneyu(df['sigma_alpha0[1]'],df['sigma_alpha0[2]']).pvalue
ax.text(0.95, 0.95, f'Mann-Whitney U test p = {pval:.3g}\nN = {len(df)}', ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel('sigma_alpha0')
ax.yaxis.grid(True)
seaborn.despine()
plt.tight_layout()
#plt.show()
plt.savefig('sigma_alpha0.png')

plt.close()
figsize = (8,6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.boxplot([df['sigma_alpha[1]'],df['sigma_alpha[2]']], labels=['Cluster 1','Cluster 2'])
pval = mannwhitneyu(df['sigma_alpha[1]'],df['sigma_alpha[2]']).pvalue
ax.text(0.95, 0.95, f'Mann-Whitney U test p = {pval:.3g}\nN = {len(df)}', ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel('sigma_alpha')
ax.yaxis.grid(True)
seaborn.despine()
plt.tight_layout()
#plt.show()
plt.savefig('sigma_alpha.png')

plt.close()
figsize = (8,6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.boxplot([df['sigma_err[1]'],df['sigma_err[2]']], labels=['Cluster 1','Cluster 2'])
pval = mannwhitneyu(df['sigma_err[1]'],df['sigma_err[2]']).pvalue
ax.text(0.95, 0.95, f'Mann-Whitney U test p = {pval:.3g}\nN = {len(df)}', ha='right', va='top', transform=ax.transAxes)
ax.set_ylabel('MA_sigma_err')
ax.yaxis.grid(True)
seaborn.despine()
plt.tight_layout()
#plt.show()
plt.savefig('MA_sigma_err.png')

"""
plt.close()
figsize = (8,6)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
vals = []
labels = []
pos = []
for di, dn in enumerate(drugnames):
    vals.extend([df['sigma_b[1,%d]'%(di+1,)].values, df['sigma_b[2,%d]'%(di+1,)].values])
    labels.append(dn)
    pos.append(3*di+0.5)    
ax.boxplot(vals)
ax.set_xticks(pos)
ax.set_xticklabels(labels)
for di, dn in enumerate(drugnames):
    pval = mannwhitneyu(df['sigma_b[1,%d]'%(di+1,)].values, df['sigma_b[2,%d]'%(di+1,)].values).pvalue
    top = max(df['sigma_b[1,%d]'%(di+1,)].max(), df['sigma_b[2,%d]'%(di+1,)].max())
    ax.plot([3*di, 3*di, 3*di+1, 3*di+1], [top+0.1,top+0.2,top+0.2,top+0.1])
    ax.text(3*di+0.5, top+0.25, f'p = {pval:.3g}\nN = {len(df)}', ha='center', va='bottom')
ax.set_ylabel('sigma_b')
ax.yaxis.grid(True)
seaborn.despine()
plt.tight_layout()
#plt.show()
plt.savefig('sigma_b.png')
"""


for di, dn in enumerate(drugnames):
    plt.close()
    figsize = (8,6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.boxplot([df['sigma_b[1,%d]'%(di+1,)],df['sigma_b[2,%d]'%(di+1,)]], labels=['Cluster 1','Cluster 2'])
    pval = mannwhitneyu(df['sigma_b[1,%d]'%(di+1,)],df['sigma_b[2,%d]'%(di+1,)]).pvalue
    ax.text(0.95, 0.95, f'Mann-Whitney U test p = {pval:.3g}\nN = {len(df)}', ha='right', va='top', transform=ax.transAxes)
    ax.set_ylabel('sigma_b_%s'%dn)
    ax.yaxis.grid(True)
    seaborn.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig('sigma_b_%s.png'%dn)


