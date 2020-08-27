#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:23:09 2020

@author: harspari
"""

import fake_sim
import policy_learning as pl

import numpy as np
import scipy
import scipy.optimize as opt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import functools
import itertools
import sys
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

def decimalToBinary(n,bits):
    a = bin(n).replace("0b", "")
    b = ''
    if len(a)!=bits:
        b = functools.reduce(lambda x,y: x+y,['0' for i in range(bits-len(a))])
    return b+a

np.random.seed(1080)

W = 1800
T = 48
lag = 2

E_init = np.array([900, 950])

a0 = 0.5
a = np.array([0.5, 0.5])

b0 = 0.2
b = np.array([0.25,0.1])

k = np.array([0.115525,0.462098])

D = np.random.binomial(1, 0.05,(48,2)) 


params  = ( a0, a, b0, b, k, lag, T, W)

# p, IIC, Dc = fake_simulator( params, E_init, D )


sim = pl.Simulator( params, E_init )
'''
sim.run_sim(T-lag,D)


IIC =  sim.E/sim.W

fig,ax = plt.subplots(1+D.shape[1],1,sharex=True,figsize=(15,10),gridspec_kw = {'height_ratios':[10]+[1 for i in range(D.shape[1])]})
ax[0].plot(np.arange(0,T),IIC, c='black',label='Observed')
ax[0].plot(np.arange(0,T),sim.p, c='#ff7f0e',label='Probability')
# ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
ax[0].legend()
ax[0].set_title('IIC Ratio')
for i in range(1,1+sim.Dc.shape[1]):
    y = sim.Dc[:,i-1]
    ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
    ax[i].set_title('Drug-Concentration-%d'%(i))
'''
estimator = pl.Estimator(sim.num_actions,method='RF')
episode_rewards, policy = pl.q_learning(sim, estimator, episode_length=1000, num_episodes=1000, discount_factor=1.0, epsilon=0.2, epsilon_decay=0.99, alpha=0.1)

fig =  plt.figure()
plt.plot(episode_rewards)

pi = lambda x: np.array(list(decimalToBinary( np.random.choice(np.arange(len(policy(x))), p=policy(x)) , bits=int(np.log2(sim.num_actions)))),dtype=int)  #learned policy
pi2 = lambda x: np.array([1,1]) #give drug all the time policy

sim.reset()
sim.run_sim_with_policy(T-2,pi)

IIC =  sim.E/sim.W

fig,ax = plt.subplots(1+2*D.shape[1],1,sharex=True,figsize=(15,20),gridspec_kw = {'height_ratios':[10]+[1 for i in range(2*D.shape[1])]})
ax[0].plot(np.arange(0,T),IIC, c='black',label='Observed')
ax[0].plot(np.arange(0,T),sim.p, c='#ff7f0e',label='Probability')
# ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
ax[0].legend()
ax[0].set_title('IIC Ratio')
for i in range(1,1+2*sim.Dc.shape[1],2):
    y = sim.Dc[:,(i-1)//2]
    y1 = sim.D[:,(i-1)//2]
    ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
    ax[i].set_title('Drug-Concentration-%d'%(i//2 + 1))
    ax[i+1].imshow(y1[np.newaxis,:], cmap="plasma", aspect="auto")
    ax[i+1].set_title('Drug-%d'%(i//2 + 1))

# def eval_regime( sim, D, reward, gamma ):
#     E = sim(D)
#     T = len(E)
#     V = np.zeros((T,))
#     V[-1] = reward(E[-1],D[-1])
#     for t in range(T-2,-1,-1):
#         V[t] = reward(E[t],D[t]) + gamma*V[t+1]
#     return V

# sim = lambda D: fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, D )[1]
# reward = lambda s,a: -s - np.linalg.norm(a)
# gamma = 0.9999

# V_curr = eval_regime(sim, fake_sim.D, reward, gamma)
# print(V_curr[0])

# V0 = lambda D: -1*eval_regime(sim, D.reshape(-1,2), reward, gamma)[0]

# res = opt.minimize( V0, fake_sim.D.reshape(-1,1), method='COBYLA' )
    
# IICopt = sim(res.x.reshape(-1,2))
# print(-V0(res.x))

# p, E, Dc = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, fake_sim.D ) 
# p1, E1, Dc1 = fake_sim.fake_simulator( fake_sim.params, fake_sim.E_init, res.x.reshape(-1,2)) 


# fig,ax = plt.subplots(1+Dc1.shape[1],1,sharex=True,figsize=(15,10),gridspec_kw = {'height_ratios':[10]+[1 for i in range(Dc1.shape[1])]})

# ax[0].plot(np.arange(0,Dc1.shape[0]),E1, '--', c='black',label='Optimal Observed')
# ax[0].plot(np.arange(0,Dc.shape[0]),E, c='black',label='Original Observed')

# ax[0].plot(np.arange(0,Dc1.shape[0]),p1, '--', c='b',label='Optimal Probability')
# ax[0].plot(np.arange(0,Dc.shape[0]),p, c='#ff7f0e',label='Original Probability')

# # ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
# ax[0].legend()
# ax[0].set_title('IIC Ratio')
# for i in range(1,1+Dc1.shape[1]):
#     y = Dc1[:,i-1]
#     ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
#     ax[i].set_title('Drug-%d'%(i))
    