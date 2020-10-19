import pickle
import numpy as np
from scipy.special import expit as sigmoid
import pystan
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


with open('../results/model_fit_lognormal.pkl', 'rb') as f:
    stan_model, fit_res = pickle.load(f)
    
with open('../results/results_lognormal.pickle', 'rb') as ff:
    res  = pickle.load(ff)
Ep = res['Ep_sim']
E = res['E']
Dscaled = res['Dscaled']
Dmax = res['Dmax']
sids = res['sids']
N = len(sids)
    
df = fit_res.to_dataframe(pars=['alpha', 'mu', 'sigma'])#'t0', 
#t0    = np.array([df['t0[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
alpha = np.array([df['alpha[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
mu    = np.array([df['mu[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)
sigma = np.array([df['sigma[%d]'%i].values for i in range(1,N+1)]).mean(axis=1)

T = 48*2
tt = np.arange(1,T+1)
P_no_drug = []
for i in range(N):
    val = alpha[i] * np.exp(-(np.log(tt)-mu[i])**2 / (2* sigma[i]**2))#+t0[i]
    val = sigmoid(val)
    P_no_drug.append(val)

time = (tt-1)*0.5
"""
plt.close()
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
for i in range(N):
    ax.plot(time, P_no_drug[i]*100)
ax.set_xlabel('Time (hour)')
ax.set_ylabel('Logit of IIC burden (%)')

plt.tight_layout()
#plt.show()
plt.savefig('lognormal_trajectory_no_drug.png')
"""

for i in range(N):
    plt.close()
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.plot(time, P_no_drug[i]*100)
    plt.tight_layout()
    plt.show()
    import pdb;pdb.set_trace()
    plt.savefig('lognormal_trajectory_no_drug.png')

