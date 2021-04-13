import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


with open('model_fit_sid2.pkl', 'rb') as f:
    model, fit = pickle.load(f)
with open('results.pickle', 'rb') as ff:
    res  = pickle.load(ff)
fit_res = res['fit_res']
sids = res['sids']

for i, fit in enumerate(tqdm(fit_res)):
    df = fit.to_dataframe(pars=['mu_a0', 'sigma_a0'])

    plt.close()
    fig = plt.figure(figsize=(9,6))
    ax1 = fig.add_subplot(211)
    ax1.plot(df['mu_a0'], c='k')
    #ax1.set_xlabel('Iteration')
    ax1.set_ylabel('mu_a0')
    ax2 = fig.add_subplot(212)
    ax2.plot(df['sigma_a0'], c='k')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('sigma_a0')
    plt.tight_layout()
    #plt.show()
    plt.savefig('stan_MCMC_trace/traceplot_%s_a0.png'%sids[i])
