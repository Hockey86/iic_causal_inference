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
for path in paths:
    res = sio.loadmat(path)
    Eobs = res['Eobs'].flatten()
    Esim = res['Esim'].flatten()
    Ebl = res['Ebaseline'].flatten()
    
    sids.append(os.path.basename(path)[2:-4])
    rmse_sim.append(np.sqrt(np.mean((Esim-Eobs)**2)))
    rmse_bl.append(np.sqrt(np.mean((Ebl-Eobs)**2)))
    
print(ttest_rel(rmse_sim, rmse_bl))

plt.close()
plt.hist(rmse_sim, bins=np.linspace(0,1,11), color='b', label='sim', rwidth=0.8, alpha=0.2)
plt.hist(rmse_bl, bins=np.linspace(0,1,11), color='k', label='bl', rwidth=0.8, alpha=0.2)
plt.legend()
plt.ylabel('Count')
plt.xlabel('RMSE')
plt.tight_layout()
plt.show()

plt.close()
plt.scatter(rmse_sim, rmse_bl)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('Sim RMSE')
plt.xlabel('Baseline RMSE')
plt.tight_layout()
plt.show()
