#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:39:28 2020

@author: harspari
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def drug_concentration(d_ts,k):
    """
    d_ts.shape = (#drug, T)
    """
    k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
    conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
    conc = conc[:,:d_ts.shape[1]]
    return conc

def fake_simulator( params, E_init, D ):
    a0, a, b0, b, k, lag, T, W = params
    A = np.zeros((T,))
    B = np.zeros((T,))
    p = np.zeros((T,))
    p[0:lag] = E_init/W
    E = np.zeros((T,))
    E[0:lag] = E_init
    Dc = drug_concentration( D.T , k ).T
    for t in range( lag, T):
        p_lag = p[t-lag:t]
        A[t] = a0 + np.dot(a,p_lag)
        B[t] = b0 + np.dot(b,Dc[t,:])
        p[t] = scipy.special.expit(A[t])*scipy.special.expit(1 - B[t])
        E[t] = np.random.binomial(W, p[t])
    return p, E/W, Dc

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

p, IIC, Dc = fake_simulator( params, E_init, D )

fig,ax = plt.subplots(1+D.shape[1],1,sharex=True,figsize=(15,10),gridspec_kw = {'height_ratios':[10]+[1 for i in range(D.shape[1])]})
ax[0].plot(np.arange(0,T),IIC, c='black',label='Observed')
ax[0].plot(np.arange(0,T),p, c='#ff7f0e',label='Probability')
# ax[0].fill_between(x=np.arange(2,T),y1=p.mean(axis=0)[2:]+p.std(axis=0)[2:],y2=p.mean(axis=0)[2:]-p.std(axis=0)[2:],alpha=0.25,color='red')
ax[0].legend()
ax[0].set_title('IIC Ratio')
for i in range(1,1+Dc.shape[1]):
    y = Dc[:,i-1]
    ax[i].imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
    ax[i].set_title('Drug-%d'%(i))