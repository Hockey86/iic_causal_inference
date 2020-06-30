#!/usr/bin/env python
# coding: utf-8

# In[1]:

import glob
import pickle
import numpy as np
import pandas as pd
import scipy 
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# Halflife of Anti-Epileptic Drugs

# In[2]:


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


# ## PK PD Model

# PK: Calculating Drug Concentration in the body

# In[4]:


def drug_concentration(d_ts,k):
    k_ts = np.array([ np.exp(-k*t) for t in range(d_ts.shape[1]) ]).T
    conc = np.array([np.convolve(d_ts[i],k_ts[i],'full') for i in range(d_ts.shape[0])])
    return conc


# PD: E’(t) = sigmoid( g(C) + n(t) + b1 + c2*nonlinearfunction(E(t-1)) ) x ( 1 - sigmoid( h(x(t)) + f(C) +b2 ) )

# In[17]:


def pharmacodynamics(args,d_conc,E0,T):
#     np.random.seed(0)
    a0 = args[0]
    a = args[1:1+E0.shape[0]]
    b0 = args[1+E0.shape[0]]
    b = np.abs(args[2+E0.shape[0]:2+E0.shape[0]+d_conc.shape[0]])
    e = np.abs(args[2+E0.shape[0]+d_conc.shape[0]])
#     g = args[3+E0.shape[0]+d_conc.shape[0]:3+E0.shape[0]+d_conc.shape[0]+C.shape[0]]
#     f = args[3+E0.shape[0]+d_conc.shape[0]+C.shape[0]:3+E0.shape[0]+d_conc.shape[0]+2*C.shape[0]]
    E = np.zeros((T,))
    E[:E0.shape[0]] = E0
    for t in range(E0.shape[0],T):
        A = a0 + np.dot(a,E[t-E0.shape[0]:t]) #+ 0*np.dot(g,C)
        B = b0 + np.dot(b,d_conc[:,t]) + np.abs(np.random.normal(0,e)) #+ 0*np.dot(f,C)
        E1 = scipy.special.expit(A)*(1-scipy.special.expit(B)) 
        E[t] = E1
    return np.clip(E,0,1)


# # Real IIC Data

# ## Read Patient's Waveforms

# In[18]:


import os
import scipy.io as io

def patient(file):
    window=900
    step=900
    df_temp = pd.DataFrame()
    if '.mat' in file:
        s = io.loadmat(file)
        human_iic_dummy = pd.get_dummies(s['human_iic'][0],dummy_na=False).rename(columns=lambda x: str(x) )
        human_iic_dummy_1hour = pd.DataFrame([np.mean(human_iic_dummy.loc[i:i+window,:],axis=0) for i in range(0,human_iic_dummy.shape[0],step)])
        
        drugs = s['drugs_weightnormalized']
        drugnames = list(map(lambda x: x.replace(' ',''),s['Dnames']))
        drugs_1hr = pd.DataFrame([ np.mean(drugs[i:i+window,:],axis=0) for i in range(0,human_iic_dummy.shape[0],step) ],columns=drugnames)
        change_drug_1hr = drugs_1hr.diff().rename(columns=lambda x: 'd_'+x)
        drugs_1hr = drugs_1hr.join(change_drug_1hr)
        
        for i in range(human_iic_dummy_1hour.shape[0]):
            if (human_iic_dummy_1hour.loc[i,:]==0).all():
                drugs_1hr.loc[i,:] = np.nan 
                human_iic_dummy_1hour.loc[i,:] = np.nan
        
        if '0.0' in human_iic_dummy_1hour.columns:
            human_iic_dummy_1hour['IIC'] = 1 - human_iic_dummy_1hour['0.0']
        else:
            human_iic_dummy_1hour['IIC'] = np.ones(len(human_iic_dummy_1hour))
        if '5.0' in human_iic_dummy_1hour.columns:
            human_iic_dummy_1hour['IIC'] = human_iic_dummy_1hour['IIC'] - human_iic_dummy_1hour['5.0']
        change_iic_1hr = human_iic_dummy_1hour.diff().rename(columns=lambda x: 'd_'+x)
        human_iic_dummy_1hour = human_iic_dummy_1hour.join(change_iic_1hr)
        
        df_temp = human_iic_dummy_1hour.join(drugs_1hr)
        df_temp['sid'] = [file.split('.')[0] for i in range(df_temp.shape[0])]
    return df_temp


# ## Leave One Out Loss

# In[29]:


def loss_loo(args, patients, sid_leave_out):
    #covariates = pd.read_csv('covariates.csv').set_index('Index').dropna(axis=1)
    k = halflife.loc['k'].to_numpy()
    loss = 0
    for pti in range(len(patients)):
        #Leave one out procedure
        p = patients[pti]
        sid = sids[pti]
        #Leave one out procedure
        if sid==sid_leave_out:
            continue
            
        np.random.seed(0)
        
        #setting up the data
        #Eobs1 = np.clip(p['IIC'].to_numpy(),0,1)
        Eobs = np.clip(p['IIC'].interpolate(method='cubic').to_numpy(),0,1)
        Dobs = p[['lacosamide', 'levetiracetam', 'midazolam', 
                        'pentobarbital','phenobarbital', 'phenytoin', 'propofol', 'valproate']].fillna(0).to_numpy().T
        T = Eobs.shape[0]
        
        #PK
        d_conc = drug_concentration(Dobs,k)[:,:T]
        
        #PD (loss)
        c = 0.001
        loss += (np.mean(np.square(np.array(pharmacodynamics(args,d_conc,Eobs[0:2],T)) - Eobs)) + c*np.linalg.norm(args))
    return loss


# ## Self Prediction Loss

# In[31]:


def loss_self(args,p,frac=0.5):
    #covariates = pd.read_csv('covariates.csv').set_index('Index').dropna(axis=1)
    k = halflife.loc['k'].to_numpy()
    
    np.random.seed(0)

    #setting up the data
    #Eobs1 = np.clip(p['IIC'].to_numpy(),0,1)
    Eobs = np.clip(p['IIC'].interpolate(method='cubic').to_numpy(),0,1)
    Dobs = p[['lacosamide', 'levetiracetam', 'midazolam', 
                    'pentobarbital','phenobarbital', 'phenytoin', 'propofol', 'valproate']].fillna(0).to_numpy().T
    T = Eobs.shape[0]

    #PK
    d_conc = drug_concentration(Dobs,k)[:,:T]

    #PD (loss)
    c = 0.001
    loss = (np.mean(np.square(np.array(pharmacodynamics(args,d_conc,Eobs[0:2],int(T*frac))) - Eobs[:int(T*frac)] )) + c*np.linalg.norm(args))
    
    return loss


# ## Function to Plot Simulated Data

# In[45]:


def plot_sim(E, Eobs, d_conc, sid):
    plt.close()
    fig,ax = plt.subplots(2,1)
    plt.title(sid)
    ax[0].plot(E)
    ax[0].plot(Eobs)
    # ax[0].plot(Erandom)
    # ax[0].plot(Enoise)
#     ax[0].plot(Eobs,'--',alpha=0.3,)
    # ax[0].fill_between(np.arange(T),y1=E+2*params[-1],y2=E-2*params[-1],alpha=0.15)

    ax[1].plot(d_conc.T,alpha=0.75)
    ax[0].legend(['Simulated','Observed'],bbox_to_anchor=(1.1, 1.05))
    ax[1].legend(halflife.columns,bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    fig.savefig('figures/'+sid+'.png')


# ## Learning Parameters for All Patients

# In[ ]:


def learn_all(patients, sids, w=100):
    Esims = []
    Ebaselines = []
    trained_sid = []
    params = []
    loss_array = []
    k = halflife.loc['k'].to_numpy()
    for pti in tqdm(range(len(patients))):
        #Leave one out procedure
        p = patients[pti]
        sid = sids[pti]
        
        #setting up the data
        #Eobs1 = np.clip(p['IIC'].to_numpy(),0,1)
        Eobs = np.clip(p['IIC'].interpolate(method='cubic').to_numpy(),0,1)
        Dobs = p[['lacosamide', 'levetiracetam', 'midazolam', 
                  'pentobarbital','phenobarbital', 'phenytoin',
                  'propofol', 'valproate']].fillna(0).to_numpy().T

        #PK
        #learn
        args0 = np.zeros((3+Eobs[0:2].shape[0]+Dobs.shape[0],))
        loss = lambda args: loss_loo(args,patients,sid) + w*loss_self(args,p)
        opt_res = scipy.optimize.minimize(loss, args0,
                        method='COBYLA',
                        options={'maxiter':1000, 'disp':False})
                        
        #if opt_res.success:
        T = Eobs.shape[0]
        d_conc = drug_concentration(Dobs,k)[:,:T]
        param = opt_res.x
        param[4:4+Dobs.shape[0]] = np.abs(param[4:4+Dobs.shape[0]])
        param[-1] = np.abs(param[-1])
        Esim = np.array(pharmacodynamics(param,d_conc,Eobs[0:2],T))
        
        # get baseline prediction
        Ebaseline = np.r_[Eobs[0:2], np.repeat(Eobs[[1]], len(Esim)-2, axis=0)]
        
        #Plot
        plot_sim(Esim, Eobs, d_conc, sid)
        io.savemat('simluations/E_%s.mat'%sid, {'Eobs':Eobs, 'Esim':Esim, 'Ebaseline':Ebaseline})
        Esims.append(Esim)
        Ebaselines.append(Ebaseline)
        trained_sid.append(sid)
        params.append(opt_res.x)
        loss_array.append(loss(opt_res.x))
            
    return Esims, trained_sid, params, loss_array


if __name__=='__main__':
    drugs_tostudy = ['lacosamide', 'levetiracetam', 'midazolam', 
                     'pentobarbital','phenobarbital', 'phenytoin', 'propofol', 'valproate']
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
        
    Esims, trained_sid, params_learned, loss_array = learn_all(patients, sids, w=100)

    """
    params_learned = np.array(params_learned)
    loss_array_2 = np.max(loss_array) - np.array(loss_array)
    # print(params_learned.mean(axis=0))
    # print(params_learned.std(axis=0))

    df_param = pd.DataFrame(params_learned,index=trained_sid,columns=['a0']+['a.%d'%(i) for i in range(1,3)]+['b0']+['b.%s'%(halflife.columns[i]) for i in range(halflife.shape[1])]+['std.noise'])
    df_param['loss'] = loss_array

    fig = plt.figure()
    sns.scatterplot(x='b.propofol',y='b.midazolam',hue='loss',data=df_param,palette='RdBu')
    plt.legend(bbox_to_anchor=(1.2, 1.05))
    fig = plt.figure()
    sns.scatterplot(x='b.propofol',y='b.levetiracetam',hue='loss',data=df_param,palette='RdBu')
    plt.legend(bbox_to_anchor=(1.2, 1.05))
    fig = plt.figure()
    sns.scatterplot(x='b.propofol',y='b0',hue='loss',data=df_param,palette='RdBu')
    plt.legend(bbox_to_anchor=(1.2, 1.05))
    fig = plt.figure()
    sns.scatterplot(x='a0',y='b0',hue='loss',data=df_param,palette='RdBu')
    plt.legend(bbox_to_anchor=(1.2, 1.05))
    fig = plt.figure()
    sns.scatterplot(x='a.1',y='a.2',hue='loss',data=df_param,palette='RdBu')
    plt.legend(bbox_to_anchor=(1.2, 1.05))
    df_param_w = pd.DataFrame((loss_array_2.reshape(-1,1)*params_learned*len(loss_array_2)/np.sum(loss_array_2)),columns=['a0']+['a.%d'%(i) for i in range(1,3)]+['b0']+['b.%s'%(halflife.columns[i]) for i in range(halflife.shape[1])]+['std.noise'])
    covariance_params = df_param_w.corr()
    # np.fill_diagonal(covariance_params.to_numpy(),np.nan)
    fig = plt.figure()
    sns.heatmap(covariance_params,cmap='RdBu',center=0.0)

    fig = plt.figure()
    sns.jointplot(x='a0',y='b0',data=df_param,kind='kde')
    sns.jointplot(x='b.propofol',y='b.midazolam',data=df_param,kind='kde')
    sns.jointplot(x='b.propofol',y='b.levetiracetam',data=df_param,kind='kde')
    sns.jointplot(x='b.propofol',y='b0',data=df_param,kind='kde')
    sns.jointplot(x='a.1',y='a.2',data=df_param,kind='kde')

    covariates = pd.read_csv('covariates.csv').set_index('Index').dropna(axis=1)

    df_join = covariates.join(df_param,how='left')
    corr_cov = df_join.corr().iloc[:22,22:]
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(corr_cov,cmap='RdBu',center=0.0)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Gender',y='b.propofol',data=df_join,palette='RdBu')
    sns.swarmplot(x='Gender',y='b.propofol',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.scatterplot(x='Age',y='b.propofol',hue='loss',data=df_join,palette='RdBu',s=100)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Hx tobacco (including ex-smokers)',y='b.propofol',data=df_join,palette='RdBu')
    sns.swarmplot(x='Hx tobacco (including ex-smokers)',y='b.propofol',data=df_join,s=10)

    # fig = plt.figure(figsize=(5,5))
    # sns.boxenplot(x='Worst GCS in 1st 24',y='b.levetiracetam',data=df_join,palette='RdBu')
    # sns.swarmplot(x='Worst GCS in 1st 24',y='b.levetiracetam',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='neuro_dx_Seizures/status epilepticus',y='b.propofol',data=df_join,palette='RdBu')
    sns.swarmplot(x='neuro_dx_Seizures/status epilepticus',y='b.propofol',data=df_join,s=10)

    # fig = plt.figure(figsize=(5,5))
    # sns.regplot(x='iGCS actual scores',y='b.propofol',data=df_join)
    # sns.swarmplot(x='iGCS actual scores',y='b.propofol',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Hx Sz /epilepsy',y='a0',data=df_join,palette='RdBu')
    sns.swarmplot(x='Hx Sz /epilepsy',y='a0',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Hx Sz /epilepsy',y='a.1',data=df_join,palette='RdBu')
    sns.swarmplot(x='Hx Sz /epilepsy',y='a.1',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Hx Sz /epilepsy',y='a.2',data=df_join,palette='RdBu')
    sns.swarmplot(x='Hx Sz /epilepsy',y='a.2',data=df_join,s=10)

    fig = plt.figure(figsize=(5,5))
    sns.boxenplot(x='Hx Sz /epilepsy',y='b0',data=df_join,palette='RdBu')
    sns.swarmplot(x='Hx Sz /epilepsy',y='b0',data=df_join,s=10)


    # In[26]:


    est_params = (params_learned).mean(axis=0)
    x_pos = np.arange(params_learned.shape[1])
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x_pos, est_params, yerr=params_learned.std(axis=0), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['a0']+['a.%d'%(i) for i in range(1,3)]+['b0']+['b.%s'%(halflife.columns[i]) for i in range(d_conc.shape[0])]+['std.noise'],rotation=85)
                       #+['g.%s'%(covariates.columns[i]) for i in range(cov.shape[0])]+['f.%s'%(covariates.columns[i]) for i in range(cov.shape[0])],rotation=85)
    fig.savefig('estimates_params_pkpd.png')
    """

