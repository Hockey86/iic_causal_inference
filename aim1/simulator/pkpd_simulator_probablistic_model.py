#!/usr/bin/env python
# coding: utf-8
import glob
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
    def __init__(self, carry_start_step=None):
        self.carry_start_step = carry_start_step
    
    def fit(self, E, D, C):
        return self
    
    def predict(self, E, D, C):
        assert len(D)==len(C)
        Ebaseline = []
        for i in range(len(E)):
            ee = np.r_[E[i][:self.carry_start_step], np.repeat(E[i][[self.carry_start_step-1]], len(E[i])-self.carry_start_step, axis=0)]
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
        for t in range(D[i].shape[1]):
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
            Bt = b0 + np.dot(b, D[i][:,t]) + np.dot(betaB, C[i])
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
    Ep, log_Pts, log_1_Pts = pharmacodynamics(a0,a1,a2,betaA,b0,b,betaB,D,C, return_P=True)
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
        N = len(E)
        ND = D[0].shape[1]
        NC = C.shape[1]
        
        # pad to have the same length
        assert [len(x) for x in E]==[len(x) for x in D]
        Ts = np.array([len(x) for x in D])
        self.maxlen = max(Ts)
        #E2 = np.array([np.r_[E[i], np.zeros(self.maxlen-len(E[i]))+np.nan] for i in range(len(E))])
        D2 = np.array([np.r_[D[i], np.zeros((self.maxlen-len(D[i]), D[i].shape[1]))+np.nan] for i in range(len(D))])
        
        self.stan_model = """
data {
    int<lower=0> N;
    int<lower=0> T;
    int<lower=0> ND;
    int<lower=0> NC;
    matrix[N,T] E;
    matrix[T,ND] D[N]
}

parameters {
    real a0;
    real a1;
    real a2;
    vector[NC] betaA;
    
    real b0;
    vector[ND] b;
    vector[NC] betaB;
    
    real theta_u;
    real theta_l;
    real p1_s;
    real p2_s;
}

transformed parameters {
    vector[T] p;
    vector[T] A;
    vector[T] B;
    p[1] = p1_s;
    p[2] = p2_s;
    for (t in 3:T) {
        A[t] = a0 + a1*p[t-1] + a2*p[t-2] + betaA;
        B[t] = b0 + dot_product(b,col(D,t));
        p[t] = (1/(1+exp(-1*A[t])))*(1 - (1/(1+exp(-1*B[t]))));
    }
}

model {
    p1_s ~ uniform(theta_l,theta_u);
    p2_s ~ uniform(theta_l,theta_u);
    for (t in 1:T) {
        E[t] ~ bernoulli(p[t]);
    }   
}"""
        
        #theano.config.mode = 'FAST_COMPILE'
        #theano.config.allow_gc = False
        
        np.random.seed(self.random_state)
        random_seed = [np.random.randint(low=0,high=100000) for _ in range(self.n_jobs)]
        
        self.D_shared = theano.shared(D2)
        self.Ts_shared = theano.shared(Ts)
        self.C_shared = theano.shared(C)
        with pm.Model() as self.model:
            # define paramters
            a0 = pm.Normal('a0', mu=0, sigma=1)
            a1 = pm.HalfNormal('a1', sigma=1)
            a2 = pm.HalfNormal('a2', sigma=1)
            betaA = pm.HalfNormal('betaA', sigma=1, shape=NC)
            
            b0 = pm.Normal('b0', mu=0, sigma=1)
            b = pm.HalfNormal('b', sigma=1, shape=ND)
            betaB = pm.Normal('betaB', sigma=1, shape=NC)
            
            # forward
            Ep = []
            #log_Pts = []
            #log_1_Pts = []
            for i in range(N):
                Ep.append([])
                #log_Pts.append([])
                #log_1_Pts.append([])
                logit_Pt = []
                for t in range(self.Ts_shared.get_value()[i]):
                    if t==0:
                        logit_Pt_1 = 0
                        logit_Pt_2 = 0
                    elif t==1:
                        logit_Pt_1 = logit_Pt[t-1]
                        logit_Pt_2 = 0
                    else:
                        logit_Pt_1 = logit_Pt[t-1]
                        logit_Pt_2 = logit_Pt[t-2]
                    At = a0 + a1*logit_Pt_1 + a2*logit_Pt_2 + betaA.dot(self.C_shared[i])
                    Bt = b0 + b.dot(self.D_shared[i][t]) + betaB.dot(self.C_shared[i])
                    log_Pt = logsigmoid_theano(At) + logsigmoid_theano(-Bt)
                    log_1_Pt = log1mexp_theano(log_Pt)
                    logit_Pt.append(log_Pt - log_1_Pt)
                    Ep[i].append(tt.exp(log_Pt))
                    #log_Pts[i].append(log_Pt)
                    #log_1_Pts[i].append(log_1_Pt)
                  
            # generate sample weights which is inverse proportional to T
            weights = np.concatenate([[1./Ts[i]]*Ts[i] for i in range(len(Ts))])
            weights = weights/weights.mean()
            
            self.W = 100
            eps = 1e-6
            Eobs = np.round(np.concatenate(E)*self.W).astype(int)
            Ep = tt.concatenate(Ep)
            Ep = pm.Deterministic('Ep', tt.clip(Ep, eps, 1-eps))
            self.potential = pm.Potential('weighted_ll', weights * pm.Binomial.dist(n=self.W, p=Ep).logp(Eobs))
             
            #step = pm.NUTS(potential=self.potential)
            self.trace = pm.sample(self.max_iter, tune=self.max_iter//2, cores=self.n_jobs, random_seed=random_seed)#, step=step)
            import pdb;pdb.set_trace()
            aa=1
            
        return self
    
    def predict(self, D, C):
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
        self.Ts_shared.set_value(np.array(Ts))
        self.C_shared.set_value(C)
        
        with self.model:
            pred = pm.sample_posterior_predictive(self.trace, samples=500)
            
        return pred['Ep']


def plot_sim(E, Eobs, d_conc, sid):
    plt.close()
    fig,ax = plt.subplots(2,1)
    plt.title(sid)
    ax[0].plot(E)
    ax[0].plot(Eobs)
    # ax[0].plot(Erandom)
    # ax[0].plot(Enoise)
    # ax[0].plot(Eobs,'--',alpha=0.3,)
    # ax[0].fill_between(np.arange(T),y1=E+2*params[-1],y2=E-2*params[-1],alpha=0.15)

    ax[1].plot(d_conc.T,alpha=0.75)
    ax[0].legend(['Simulated','Observed'],bbox_to_anchor=(1.1, 1.05))
    ax[1].legend(halflife.columns,bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    fig.savefig('figures/'+sid+'.png')


def learn_all(patients, C):
    k = halflife.loc['k'].to_numpy()
    E = []
    D = []
    sids = []
    for p in patients:
        E.append(np.clip(p['IIC'].interpolate(method='cubic').to_numpy(),0,1))
        #PK
        Ddose = p[['lacosamide', 'levetiracetam', 'midazolam', 
                  'pentobarbital','phenobarbital', 'phenytoin',
                  'propofol', 'valproate']].fillna(0).to_numpy().T
        D.append(drug_concentration(Ddose,k)[:,:len(E[-1])].T)
        sids.append(os.path.basename(p.sid.iloc[0]))
    #E = np.array(E)
    #D = np.array(D)
    #sids = np.array(sids)
    
    Esims = []
    Ebaselines = []
    trained_sid = []
    params = []
    loss_array = []
    for pti in tqdm(range(len(sids))):
        #Leave one out procedure
        sid = sids[pti]
        
        Etr = [E[x][:20] for x in range(len(sids)) if sids[x]!=sid]
        Dtr = [D[x][:20] for x in range(len(sids)) if sids[x]!=sid]
        Ctr = np.array([C[x] for x in range(len(sids)) if sids[x]!=sid])
        Ete = E[sids.index(sid)]
        Dte = D[sids.index(sid)]
        Cte = C[[sids.index(sid)]]
        
        Cmean = Ctr.mean(axis=0)
        Cstd = Ctr.std(axis=0)
        Ctr = (Ctr-Cmean)/Cstd
        Cte = (Cte-Cmean)/Cstd
        
        #fit simulator
        #model = IICSimulatorMLE()
        model = IICSimulatorBayesian(max_iter=50, n_jobs=8, random_state=random_state)
        model.fit(Etr, Dtr, Ctr)
        Esim = model.predict([Dte], Cte)[0]
        
        # get baseline prediction
        model = IICSimulatorCarryForward(carry_start_step=2)
        model.fit(Etr, Dtr, Ctr)
        Ebaseline = model.predict([Ete[:2]], [Dte], Cte)[0]
        
        #Plot
        plot_sim(Esim, Ete, Dte, sid)
        sio.savemat('simluations/E_%s.mat'%sid, {'Eobs':Ete, 'Esim':Esim, 'Ebaseline':Ebaseline})
        Esims.append(Esim)
        Ebaselines.append(Ebaseline)
        trained_sid.append(sid)
        params.append(opt_res.x)
        loss_array.append(opt_res.func)
            
    return Esims, trained_sid, params, loss_array


if __name__=='__main__':
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                     'pentobarbital','phenobarbital', 'phenytoin',
                     'propofol', 'valproate']
    paths = glob.glob('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output/sid*.mat')
    
    """
    patients = []
    sids = []
    for path in tqdm(paths):
        p = patient(path)
        if p[drugs_tostudy].fillna(0).values.max()>0:
            patients.append(p)
            sids.append(os.path.basename(path).replace('.mat',''))
    """
    with open('input_data.pickle', 'rb') as ff:
        patients, sids = pickle.load(ff)
    patients = patients[:20]
    sids = sids[:20]
    
    Cnames = ['Age', 'Hx CVA (including TIA)',
        'Hx Sz /epilepsy', 
        'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)',
        'neuro_dx_Seizures/status epilepticus']
    
    cov = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output/covariates.csv')
    cov_sids = list(cov.Index)
    ids = [cov_sids.index(x) for x in sids]
    cov = cov.iloc[ids].reset_index(drop=True)
    cov = cov[Cnames]
    
    C = cov.values.astype(float)
    #C = C[:,np.nanstd(C, axis=0)>0.1]
    C = KNNImputer(n_neighbors=10).fit_transform(C)
    Esims, trained_sid, params_learned, loss_array = learn_all(patients, C)

