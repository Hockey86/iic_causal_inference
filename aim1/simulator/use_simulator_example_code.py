import pickle
import numpy as np
from myfunctions import preprocess, get_pk_k
from simulator import Simulator


W = 300
model_type = 'cauchy_expit_ARMA16'
AR_p = int(model_type[-2:-1])
MA_q = int(model_type[-1:])
max_iter = 1000
stan_path = 'stan_models/model_%s.stan'%model_type
model_path = 'results/model_fit_%s_iter%d.pkl'%(model_type, max_iter)
data_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'

# get stuff
with open('data_to_fit.pickle', 'rb') as f:
    res = pickle.load(f)
sids = res['sids']
Dmax = res['Dmax']
Dname = res['Dname']
cluster = res['cluster']
Ncluster = len(set(cluster))
PK_K = get_pk_k()

# load this patient data
sid = 'sid942'
sid_index = sids.index(sid)
Pobs, Pname, D, Dname, C, Cname, spec, freq = preprocess(sid, data_dir, PK_K, W, Dname)

# remove the first K steps so that the first AR_p steps for initialization does not have NaN
K = np.where(~np.isnan(Pobs))[0][0]
Pobs = Pobs[K:]
D = D[K:]
spec = spec[K:]

# scale drug concentration, as consistent with training
D = D/Dmax
# get cluster of this patient, ideally it should be from cluster_model.predict(C)
cluster = cluster[sid_index]

# load model
simulator = Simulator(stan_path, W, T0=[AR_p, MA_q], max_iter=max_iter)
simulator.load_model(model_path)

# do prediction
# since we deal with one patient here, make D a list since Simulator.predict expects a list
D = [D]
cluster = [cluster]
Pstart = [Pobs[:AR_p]]
simulator.random_state = 2020  # IMPORTANT, if want different results, set random_state
Psim = simulator.predict(D, cluster, Ncluster=Ncluster,
                         sid_index=[sid_index],
                         Pstart=Pstart, posterior_mean=True)
Psim = [x[0] for x in Psim]  # since posterior_mean is True, it returns 1 posterior draw using the mean, take [0] to get it
Psim = Psim[0] # since we deal with one patient here

# finally, Psim.shape = (len(D),)
print('D.shape=', D[0].shape)
print('Psim.shape=', Psim.shape)


import matplotlib.pyplot as plt
plt.plot(Pobs, c='k',label='actual');plt.plot(Psim,c='r',label='sim');plt.legend();plt.show()

