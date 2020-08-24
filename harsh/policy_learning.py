#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:31:00 2020

@author: harspari
"""

import fake_sim
import numpy as np
import scipy
import scipy.optimize as opt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def eval_regime( sim, D, reward, gamma ):
    E = sim(D)
    T = len(E)
    V = np.zeros((T,))
    V[-1] = reward(E[-1],D[-1])
    for t in range(T-2,-1,-1):
        V[t] = reward(E[t],D[t]) + gamma*V[t+1]
    return V

sim = lambda D: fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, D )[1]
reward = lambda s,a: -s - np.linalg.norm(a)
gamma = 0.9999

V_curr = eval_regime(sim, fake_sim.D, reward, gamma)
print(V_curr[0])

V0 = lambda D: -1*eval_regime(sim, D.reshape(-1,2), reward, gamma)[0]

res = opt.minimize( V0, fake_sim.D.reshape(-1,1), method='COBYLA' )
    
IICopt = sim(res.x.reshape(-1,2))
print(-V0(res.x))

p, E, Dc = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, fake_sim.D ) 
p1, E1, Dc1 = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, res.x.reshape(-1,2)) 


fig,ax = plt.subplots(1+Dc1.shape[1],1,sharex=True,figsize=(15,10),gridspec_kw = {'height_ratios':[10]+[1 for i in range(Dc1.shape[1])]})

ax[0].plot(np.arange(0,Dc1.shape[0]),E1, '--', c='black',label='Optimal Observed')
ax[0].plot(np.arange(0,Dc.shape[0]),E, c='black',label='Original Observed')

ax[0].plot(np.arange(0,Dc1.shape[0]),p1, '--', c='b',label='Optimal Probability')
ax[0].plot(np.arange(0,Dc.shape[0]),p, c='#ff7f0e',label='Original Probability')

# ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
ax[0].legend()
ax[0].set_title('IIC Ratio')
for i in range(1,1+Dc1.shape[1]):
    y = Dc1[:,i-1]
    ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
    ax[i].set_title('Drug-%d'%(i))