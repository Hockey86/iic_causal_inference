import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


with open('results.pickle', 'rb') as ff:
    res  = pickle.load(ff)

W = 900
Ep = res['Ep_sim'][0]
Eobs = res['E']
D = res['D']
sids = res['sids']
ND = D[0].shape[-1]
Dnames = ['lacosamide', 'levetiracetam', 'midazolam', 
          'pentobarbital','phenobarbital',# 'phenytoin',
          'propofol', 'valproate']

for si, sid in enumerate(tqdm(sids)):
    tt = np.arange(len(Ep[si]))/2  # 1 step is half an hour
    Pobs = np.array(Eobs[si]).astype(float)
    Pobs[Pobs==-1] = np.nan
    Pobs = Pobs/W*100
    
    plt.close()
    fig = plt.figure(figsize=(9,6))
    
    ax1 = fig.add_subplot(211)
    ax1.plot(tt, np.array(Ep[si])*100, c='r', label='simulated')
    ax1.plot(tt, Pobs, c='k', label='observed')
    ax1.legend()
    #ax1.set_xlabel('time (h)')
    ax1.set_ylabel('IIC burden (%)')
    ax1.set_ylim([-2,102])
    
    ax2 = fig.add_subplot(212)
    for di in range(ND):
        if np.max(D[si][:,di])>0:
            ax2.plot(tt, D[si][:,di], label=Dnames[di])
    ax2.legend()#ncol=2
    ax2.set_xlabel('time (h)')
    ax2.set_ylabel('Drug concentration')
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('simulation_global_figures/%s.png'%sids[si])
