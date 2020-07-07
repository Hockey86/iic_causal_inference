#!/usr/bin/env python
# coding: utf-8
import glob
from itertools import groupby
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from scipy.special import logit
import scipy.io as sio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.impute import KNNImputer
#import pystan
import theano
import theano.tensor as tt
import pymc3 as pm


random_state = 2020

# Halflife of Anti-Epileptic Drugs

halflife = pd.DataFrame({
    'lacosamide':[13],
    'levetiracetam':[6],
    'midazolam':[1.5],
    'pentobarbital':[15],
    'phenobarbital':[53],
    'phenytoin':[22],
    'propofol':[1.5],
    'valproate':[8]
    },index=['t1/2'])

halflife = halflife.append(np.log(2) / halflife.rename(index={'t1/2':'k'}))


def patient(file):
    window = 900
    step   = 900
    if '.mat' in file:
        s = sio.loadmat(file)
        human_iic = s['human_iic'][0].astype(float)
        spike = s['spike'][0].astype(float)
        drugs = s['drugs_weightnormalized'].astype(float)
        artifact = s['artifact'][0].astype(float)
        human_iic[artifact==1] = np.nan
        spike[artifact==1] = np.nan

        drugnames = list(map(lambda x: x.strip(), s['Dnames']))
        drugs_window = np.array([ np.mean(drugs[i:i+window],axis=0) for i in range(0,len(drugs),step) ])

        sz_burden = (human_iic==1).astype(float)
        sz_burden[np.isnan(human_iic)] = np.nan
        sz_burden_window = [np.nanmean(sz_burden[i:i+window]) for i in range(0, len(sz_burden),step)]

        iic_burden = np.in1d(human_iic, [1,2,3,4]).astype(float)
        iic_burden[np.isnan(human_iic)] = np.nan
        iic_burden_window = [np.nanmean(iic_burden[i:i+window]) for i in range(0, len(iic_burden),step)]

        spike_rate_window = [np.nanmean(spike[i:i+window]) for i in range(0, len(spike),step)]

        df = pd.DataFrame(data=np.c_[sz_burden_window, iic_burden_window, spike_rate_window, drugs_window],
                          columns=['sz_burden', 'iic_burden', 'spike_rate']+drugnames)
    return df


def logsigmoid_numpy(x):
    """
    Computes the log(sigmoid(x))
    http://fa.bianp.net/blog/2019/evaluate_logistic/#sec2
    """
    x = np.array(x)
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    if type(x)==float or x.ndim==0:
        out = float(out)
    return out


def logsigmoid_theano(x):
    return -tt.nnet.softplus(-x)


def log1mexp_numpy(x):
    """
    Compute log(1-exp(x))
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp_numpy-note.pdf
    """
    a0 = -np.log(2)
    x = np.array(x)
    out = np.zeros_like(x)
    idx = x >= a0
    out[idx] = np.log(-np.expm1(x[idx]))
    idx = x < a0
    out[idx] = np.log1p(-np.exp(x[idx]))
    if type(x)==float or x.ndim==0:
        out = float(out)
    return out


def log1mexp_theano(x):
    """
    Compute log(1-exp(x))
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp_numpy-note.pdf
    """
    return tt.switch(tt.gt(x, -0.683), tt.log(-tt.expm1(x)), tt.log1p(-tt.exp(x)))


def drug_concentration(d_ts,k):
    k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
    conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
    return conc


class IICSimulatorCarryForward(object):
    def __init__(self, carry_start=None):
        self.carry_start = carry_start
    
    def fit(self, E, D, C):
        return self
    
    def predict(self, E, D, C):
        assert len(D)==len(C)
        Ebaseline = []
        for i in range(len(E)):
            # if there is nan, use the closest non-nan location
            notnan_ids = np.where(~np.isnan(E[i]))[0]
            closest_notnan_id = np.argmin(np.abs(notnan_ids-self.carry_start))
            carry_start = notnan_ids[closest_notnan_id]
            ee = np.r_[E[i][:carry_start], np.repeat(E[i][[carry_start-1]], len(D[i])-carry_start, axis=0)]
            Ebaseline.append(ee)
        return Ebaseline


def pharmacodynamics(a0,a1,a2,betaA,b0,b,betaB,D,C, return_P=False):
    E = []
    log_Pts = []
    log_1_Pts = []
    for i in range(len(D)):
        E.append([])
        log_Pts.append([])
        log_1_Pts.append([])
        logit_Pt = []
        for t in range(len(D[i])):
            if t==0:
                logit_Pt_1 = 0
                logit_Pt_2 = 0
            elif t==1:
                logit_Pt_1 = logit_Pt[t-1]
                logit_Pt_2 = 0
            else:
                logit_Pt_1 = logit_Pt[t-1]
                logit_Pt_2 = logit_Pt[t-2]
            At = a0 + a1*logit_Pt_1 + a2*logit_Pt_2 + np.dot(betaA, C[i])
            Bt = b0 + np.dot(b, D[i][t]) + np.dot(betaB, C[i])
            log_Pt = logsigmoid_numpy(At) + logsigmoid_numpy(-Bt)
            log_1_Pt = log1mexp_numpy(log_Pt)
            logit_Pt.append(log_Pt - log_1_Pt)
            E[i].append(np.exp(log_Pt))
            log_Pts[i].append(log_Pt)
            log_1_Pts[i].append(log_1_Pt)

    for i in range(len(D)):
        E[i] = np.array(E[i])
        log_Pts[i] = np.array(log_Pts[i])
        log_1_Pts[i] = np.array(log_1_Pts[i])
        
    if return_P:
        return E, log_Pts, log_1_Pts
    else:
        return E
    
    
def myloss(x,E,D,C):
    a0, a1, a2, betaA, b0, b, betaB = get_params(x,D,C)
    _, log_Pts, log_1_Pts = pharmacodynamics(a0,a1,a2,betaA,b0,b,betaB,D,C, return_P=True)
    loss = np.mean([np.mean(- E[i]*log_Pts[i] - (1-E[i])*log_1_Pts[i]) for i in range(len(E))])
    return loss


class IICSimulatorMLE(object):
    def _get_params(self, x):
        a0 = x[0]
        a1 = x[1]
        a2 = x[2]
        betaA = x[3:3+self.dim_C]
        b0 = x[3+self.dim_C]
        b = x[3+self.dim_C+1:3+C.shape[1]+1+self.dim_D]
        betaB = x[-self.dim_C:]
        return a0, a1, a2, betaA, b0, b, betaB
    
    def fit(self, E, D, C):
        assert len(E)==len(D)==len(C)
        self.dim_C = C.shape[1]
        self.dim_D = D[0].shape[1]
        
        x0 = np.r_[[0, 0.001, 0.001],  # a0, a1, a2
                   [0.001]*self.dim_C, # betaA
                   [0], # b0
                   [0.001]*self.dim_D, #b
                   [0.001]*self.dim_C,] #betaB
        opt_res = minimize(myloss, x0, args=(Etr,Dtr,Ctr),
                        method='COBYLA',
                        options={'maxiter':1000, 'disp':False})
                        
        self.a0, self.a1, self.a2, self.betaA, self.b0, self.b, self.betaB = self._get_params(self.opt_res.x)
        return self
    
    def predict(self, D, C):
        assert len(D)==len(C)
        Esim = pharmacodynamics(self.a0, self.a1, self.a2,
                                self.betaA, self.b0, self.b, self.betaB, D, C)
        return Esim
        
        
class IICSimulatorBayesian(object):
    def __init__(self, max_iter=100, n_jobs=1, random_state=None):
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit(self, E, D, C):
        assert len(E)==len(D)==len(C)
        T = E.shape[1]
        N = len(E)
        ND = D.shape[2]
        NC = C.shape[1]
        
        #theano.config.mode = 'FAST_COMPILE'
        theano.config.allow_gc = False
        
        np.random.seed(self.random_state)
        random_seed = [np.random.randint(low=0,high=100000) for _ in range(self.n_jobs)]
        
        #self.E_shared = theano.shared(E2)
        #self.D_shared = theano.shared(D2)
        #self.Ts_shared = theano.shared(Ts)
        #self.C_shared = theano.shared(C)
        with pm.Model() as self.model:
            # define paramters
            
            a0 = pm.Normal('a0', mu=0, sigma=1)
            a1 = pm.Normal('a1', mu=0, sigma=1)
            a2 = pm.Normal('a2', mu=0, sigma=1)
            betaA = pm.HalfNormal('betaA', sigma=1, shape=NC)
            
            b0 = pm.Normal('b0', mu=0, sigma=1)
            b = pm.HalfNormal('b', sigma=1, shape=ND)
            betaB = pm.Normal('betaB', sigma=1, shape=NC)
            
            # forward
            Pt = []
            logit_Pt = []
            for t in range(T):
                if t==0:
                    logit_Pt_1 = 0  #TODO try with and without logit
                    logit_Pt_2 = 0
                elif t==1:
                    logit_Pt_1 = logit_Pt[t-1]
                    logit_Pt_2 = 0
                else:
                    logit_Pt_1 = logit_Pt[t-1]
                    logit_Pt_2 = logit_Pt[t-2]
                At = a0 + a1*logit_Pt_1 + a2*logit_Pt_2 + tt.dot(C,betaA)
                Bt = b0 + tt.dot(D[:,t],b) + tt.dot(C,betaB)
                log_Pt = logsigmoid_theano(At) + logsigmoid_theano(-Bt)
                log_1_Pt = log1mexp_theano(log_Pt)
                logit_Pt.append(log_Pt - log_1_Pt)
                Pt.append(tt.exp(log_Pt))
                  
            self.W = 900
            eps = 1e-6
            Pt = tt.concatenate(Pt)
            Pt = pm.Deterministic('Pt', tt.clip(Pt, eps, 1-eps))
            Eobs = tt.round(E.T.flatten()*self.W).astype('int64')
            pm.Binomial('ll', n=self.W, p=Pt, observed=Eobs)
             
            #step = pm.NUTS(potential=self.potential)
            self.trace = pm.sample(self.max_iter, tune=self.max_iter//2, cores=self.n_jobs, random_seed=random_seed)#, step=step)
        
            self.var_names = ['a0', 'a1', 'a2', 'betaA', 'b0', 'b', 'betaB']
            self.Nsample = 1000
            pred = pm.sample_posterior_predictive(self.trace, samples=self.Nsample, var_names=self.var_names)
            
        for vn in self.var_names:
            setattr(self, vn, np.mean(pred[vn], axis=0))
            setattr(self, vn+'_posterior', pred[vn])
            setattr(self, vn+'_lb', np.percentile(pred[vn], 2.5, axis=0))
            setattr(self, vn+'_ub', np.percentile(pred[vn], 97.5, axis=0))
            
        return self
    
    def predict(self, D, C, verbose=True):
        #TODO model_factory
        """
        D2 = []
        Ts = []
        for i in range(len(D)):
            if len(D[i])<self.maxlen:
                D2.append(np.r_[D[i], np.zeros((self.maxlen-len(D[i]), D[i].shape[1]))+np.nan])
                Ts.append(len(D[i]))
            else:
                D2.append(D[i][:self.maxlen])
                Ts.append(self.maxlen)
        self.D_shared.set_value(np.array(D2))
        self.E_shared.set_value(np.zeros((len(D2), self.maxlen)))
        self.Ts_shared.set_value(np.array(Ts))
        self.C_shared.set_value(C)
        """
        Esim = []
        np.random.seed(self.random_state)
        for i in tqdm(range(self.Nsample), disable=not verbose):
            a0 = self.a0_posterior[i]
            a1 = self.a1_posterior[i]
            a2 = self.a2_posterior[i]
            betaA = self.betaA_posterior[i]
            b0 = self.b0_posterior[i]
            b = self.b_posterior[i]
            betaB = self.betaB_posterior[i]
            
            # get Pt which is the parameter of the binomial distribution
            Pt = pharmacodynamics(a0,a1,a2,betaA,b0,b,betaB,D,C)
            # sample from the distribution
            Ep = [np.random.binomial(self.W, x)/self.W for x in Pt]

            Esim.append(Ep)
        return np.array(Esim)


def plot_sim(E, Eobs, d_conc, sid):
    plt.close()
    fig,ax = plt.subplots(2,1)
    plt.title(sid)
    ax[0].plot(E[:,0], color='r', label='Simulated')
    ax[0].plot(E[:,1], color='r')
    ax[0].plot(E[:,2], color='r')
    ax[0].plot(Eobs, color='k', label='Observed')
    # ax[0].plot(Enoise)
    # ax[0].fill_between(np.arange(T),y1=E+2*params[-1],y2=E-2*params[-1],alpha=0.15)
    ax[0].legend(bbox_to_anchor=(1.1, 1.05))

    ax[1].plot(d_conc,alpha=0.75)
    ax[1].legend(halflife.columns, bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    fig.savefig('figures/'+sid+'.png')


def preprocess(sids, patients, C):
    sids_fixlen = []
    E_fixlen = []
    D_fixlen = []
    C_fixlen = []
    sids_varlen = []
    E_varlen = []
    D_varlen = []
    C_varlen = []

    PK_K = halflife.loc['k'].to_numpy()
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                    'pentobarbital','phenobarbital', 'phenytoin',
                    'propofol', 'valproate']
    response_tostudy = ['iic_burden']
    window = 10  # 5h
    step = 5     # 2.5h

    for pi, p in enumerate(patients):
        #E_pt = np.clip(p['IIC'].interpolate(method='cubic').to_numpy(),0,1)
        E_pt = p[response_tostudy].values.flatten()
        
        #PK
        Ddose = p[drugs_tostudy].fillna(0).to_numpy().T
        D_pt = drug_concentration(Ddose, PK_K)[:,:len(E_pt)].T

        E_varlen.append(E_pt)
        D_varlen.append(D_pt)
        C_varlen.append(C[pi])
        sids_varlen.append(sids[pi])

        # generate fixed-length segments from chunks of not-nan
        E_pt_nan = np.isnan(E_pt)
        cc = 0
        for k, l in groupby(E_pt_nan):
            ll = len(list(l))
            if not k and ll>=window:
                for j in np.arange(cc, cc+ll-window+1, step):
                    E_fixlen.append(E_pt[j:j+window])
                    D_fixlen.append(D_pt[j:j+window])
                    C_fixlen.append(C[pi])
                    sids_fixlen.append(sids[pi])
            cc += ll

    sids_fixlen = np.array(sids_fixlen)
    E_fixlen = np.array(E_fixlen)
    D_fixlen = np.array(D_fixlen)
    C_fixlen = np.array(C_fixlen)
    sids_varlen = np.array(sids_varlen)
    #E_varlen = np.array(E_varlen)
    #D_varlen = np.array(D_varlen)
    C_varlen = np.array(C_varlen)

    return sids_fixlen, E_fixlen, D_fixlen, C_fixlen, sids_varlen, E_varlen, D_varlen, C_varlen


def learn_all(sids_fixlen, E_fixlen, D_fixlen, C_fixlen, sids_varlen, E_varlen, D_varlen, C_varlen):
    Esims = []
    Ebaselines = []
    models = []

    #Leave one out procedure
    for pti in tqdm(range(len(sids_varlen))):
        sid = sids_varlen[pti]
        trids = sids_fixlen!=sid
        teid = pti
        
        # split into training and testing
        Etr = E_fixlen[trids]
        Dtr = D_fixlen[trids]
        Ctr = C_fixlen[trids]
        Ete = E_varlen[teid]
        Dte = D_varlen[teid]
        Cte = C_varlen[[teid]]

        # standardize features
        Cmean = Ctr.mean(axis=0)
        Cstd = Ctr.std(axis=0)
        Ctr = (Ctr-Cmean)/Cstd
        Cte = (Cte-Cmean)/Cstd
        
        # PD
        
        # get baseline prediction
        model = IICSimulatorCarryForward(carry_start=2)
        model.fit(Etr, Dtr, Ctr)
        Ebaseline = model.predict([Ete], [Dte], Cte)[0]
        
        #fit simulator
        #model = IICSimulatorMLE()
        model = IICSimulatorBayesian(max_iter=500, n_jobs=4, random_state=random_state)
        model.fit(Etr, Dtr, Ctr)
        import pdb;pdb.set_trace()
        Esim_posterior = model.predict([Dte], Cte)
        Esim = np.mean(Esim_posterior, axis=0)[0]
        Esim_lb, Esim_ub = np.percentile(Esim_posterior, (2.5,97.5), axis=0)
        Esim_lb = Esim_lb[0]
        Esim_ub = Esim_ub[0]
        Esim = np.c_[Esim, Esim_lb, Esim_ub]
        
        #Plot
        plot_sim(Esim, Ete, Dte, sid)
        sio.savemat('simluations/E_%s.mat'%sid, {'Eobs':Ete, 'Esim':Esim, 'Ebaseline':Ebaseline})
        Esims.append(Esim)
        Ebaselines.append(Ebaseline)
        models.append(model)
            
    return Esims, Ebaselines, models


if __name__=='__main__':
    ## load data
    """
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                     'pentobarbital','phenobarbital', 'phenytoin',
                     'propofol', 'valproate']
    paths = glob.glob('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output/sid*.mat')
    
    patients = []
    sids = []
    for path in tqdm(paths):
        p = patient(path)
        if p[drugs_tostudy].fillna(0).values.max()>0:
            patients.append(p)
            sids.append(os.path.basename(path).replace('.mat',''))
    """
    with open('input_data.pickle', 'rb') as ff:
        sids, patients, cov = pickle.load(ff)
    #patients = patients[:20]
    #sids = sids[:20]
    
    ## preprocessing
    Cnames = ['Age']
    C = cov[Cnames].values.astype(float)
    # fill missing value
    if np.any(np.isnan(C)):
        #C = C[:,np.nanstd(C, axis=0)>0.1]
        C = KNNImputer(n_neighbors=10).fit_transform(C)

    sids_fixlen, E_fixlen, D_fixlen, C_fixlen, sids_varlen, E_varlen, D_varlen, C_varlen =\
                preprocess(sids, patients, C)
    
    Esims, Ebaselines, models = learn_all(sids_fixlen, E_fixlen, D_fixlen, C_fixlen, sids_varlen, E_varlen, D_varlen, C_varlen)

    # E_varlen, Esims, Ebaselines

