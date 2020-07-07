import glob
import os
import numpy as np
import scipy.io as sio
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


paths = glob.glob('simluations/E_sid*.mat')
sids = []
rmse_sim = []
rmse_bl = []
a0 = []
a1 = []
a2 = []
betaA = []
b0 = []
b = []
betaB = []
for path in paths:
    res = sio.loadmat(path)
    Eobs = res['Eobs'].flatten()
    Esim = res['Esim'][:,0]
    Ebl = res['Ebaseline'].flatten()
    
    a0.append(res['a0_posterior'].mean())
    a1.append(res['a1_posterior'].mean())
    a2.append(res['a2_posterior'].mean())
    betaA.append(res['betaA_posterior'].mean(axis=0))
    b0.append(res['b0_posterior'].mean())
    b.append(res['b_posterior'].mean(axis=0))
    betaB.append(res['betaB_posterior'].mean(axis=0))
    
    sids.append(os.path.basename(path)[2:-4])
    rmse_sim.append(np.sqrt(np.nanmean((Esim-Eobs)**2)))
    rmse_bl.append(np.sqrt(np.nanmean((Ebl-Eobs)**2)))
    
print(ttest_rel(rmse_sim, rmse_bl))
print('a0', np.mean(a0), np.std(a0))
print('a1', np.mean(a1), np.std(a1))
print('a2', np.mean(a2), np.std(a2))
print('betaA', np.mean(betaA, axis=0), np.std(betaA, axis=0))
print('b0', np.mean(b0), np.std(b0))
print('b', np.mean(b, axis=0), np.std(b, axis=0))
print('betaB', np.mean(betaB, axis=0), np.std(betaB, axis=0))
print(rmse_sim, rmse_bl)

"""
plt.close()
plt.hist(rmse_sim, bins=np.linspace(0,1,11), color='b', label='Simulation', rwidth=0.8, alpha=0.2)
plt.hist(rmse_bl, bins=np.linspace(0,1,11), color='k', label='Baseline', rwidth=0.8, alpha=0.2)
plt.legend()
plt.ylabel('Count')
plt.xlabel('RMSE')
plt.tight_layout()
plt.show()
"""

plt.close()
plt.scatter(rmse_sim, rmse_bl)
plt.plot([0,1], [0,1], ls='--')
plt.xlabel('RMSE from simulation')
plt.ylabel('RMSE from Baseline')
plt.tight_layout()
plt.show()
