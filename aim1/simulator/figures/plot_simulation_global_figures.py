import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


W = 900
Dnames = ['lacosamide', 'levetiracetam', 'midazolam', 
          'pentobarbital','phenobarbital',# 'phenytoin',
          'propofol', 'valproate']
ND = len(Dnames)
models = ['AR2', 'AR1', 'baseline']#'lognormal', 

for model in models:
    print(model)
    
    figure_dir = 'simulation_global_figures/%s'%model
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    with open('../results/results_%s.pickle'%model, 'rb') as ff:
        res  = pickle.load(ff)

    Ep = res['Ep_sim']
    E = res['E']
    Dscaled = res['Dscaled']
    Dmax = res['Dmax']
    sids = res['sids']

    for si, sid in enumerate(tqdm(sids)):
        tt = np.arange(len(E[si]))/2  # 1 step is half an hour
        P = np.array(E[si]).astype(float)
        P[P==-1] = np.nan
        P = P/W*100
        
        plt.close()
        fig = plt.figure(figsize=(9,6))
        
        ax1 = fig.add_subplot(211)
        ax1.plot(tt, np.mean(Ep[si], axis=0)*100, c='r', label='simulated')
        ax1.plot(tt, np.percentile(Ep[si],2.5,axis=0)*100, c='r')
        ax1.plot(tt, np.percentile(Ep[si],97.5,axis=0)*100, c='r')
        ax1.plot(tt, P, c='k', label='observed')
        ax1.legend()
        #ax1.set_xlabel('time (h)')
        ax1.set_ylabel('IIC burden (%)')
        ax1.set_ylim([-2,102])
        
        ax2 = fig.add_subplot(212)
        for di in range(ND):
            if np.max(Dscaled[si][:,di])>0:
                ax2.plot(tt, Dscaled[si][:,di]*Dmax[di], label=Dnames[di])
        ax2.legend()#ncol=2
        ax2.set_xlabel('time (h)')
        ax2.set_ylabel('Drug concentration')
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_dir, '%s.png'%sids[si]))
