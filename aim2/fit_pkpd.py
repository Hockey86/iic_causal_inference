from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.optimize import least_squares, minimize
from scipy.io import loadmat

def prepareXy(df):
    Cnames = [x for x in df.columns if x.startswith('C_')]  # C is not used since everyone has their own model
    Dnames = [x for x in df.columns if x.startswith('D_')]
    E = []
    D = []
    C = []
    sids = df.SID.unique()
    ids = defaultdict(list)
    for si, sid in enumerate(df.SID):
        ids[sid].append(si)
    for sid in tqdm(sids):
        df_ = df.iloc[ids[sid]]

        # smooth IIC
        iic = df_['IIC'].values
        notnan_ids = ~np.isnan(iic)
        tt = np.arange(len(df_))[notnan_ids]
        yy = np.clip(iic[notnan_ids], 1e-2, 1-1e-2)
        yy = logit(yy)
        func = UnivariateSpline(tt, yy, k=3, s=10)
        iic_smoothed = func(np.arange(len(df_)))
        iic_smoothed = expit(np.clip(savgol_filter(iic_smoothed,11,3), -20,20))
        iic_smoothed[~notnan_ids] = np.nan
        
        E.append(iic_smoothed)
        D.append(df_[Dnames].values)
        C.append(df_[Cnames].iloc[0])
    C = np.array(C)

    return E, D, C, sids, Dnames, Cnames

def PK(W,halflife):
    D = np.zeros_like(W)
    for t in range(W.shape[0]):
        for i in range(t+1):
            D[t,:] = D[t,:] + W[t-i,:] * np.exp(-i * np.log(2)/halflife)
    return D

def fit_PD(E,D): 
    D_ = D.values
    # if np.sum(np.sum(D))==0:
    #     return [np.nan,np.nan,np.nan] * D_.shape[1] + [np.nan,np.nan,np.nan]
    
    pd_params = [0,1,0.5] * D_.shape[1] + [1, 1, 1]
    
    #remove nonzero idx
    E_ = E.values

    def Multi_Hill(pd_params): 
        error = 0
        PD_vals = np.zeros_like(E_)
        for t in range(0, E.shape[0]):
            PD_val = 1 - (np.exp( -1*((np.log(t+1) - pd_params[-2])**2) / ( 2*(pd_params[-1])**2 ) ))
            # PD_val = 1 - (pd_params[-3] * np.exp( -1*((t+1) - pd_params[-2])/ ( 2*(pd_params[-1])**2 ) ) )
            for i in range(0, D.shape[1]) :
                if E_[t]>0:
                    num1 = D_[t,i]
                    num2 = pd_params[3*i+1]
                    den1 = pd_params[3*i]
                    den2 = D_[t,i]
                    PD_val = PD_val + ( num2 * ( num1**(pd_params[3*i + 2]) )/( den1**(pd_params[3*i + 2]) + den2**(pd_params[3*i + 2]) ) )
            # PD_val = np.minimum(1,np.maximum(0, PD_val))
            PD_vals[t] = pd_params[-3] * PD_val
        error = np.linalg.norm( ( (1-E_) - PD_vals ), ord=2 ) + 1e+2*np.sum(PD_vals<0) +  1e+2*np.sum(PD_vals>1)
        return error       
    
    lb = [0,0.1,0] *D_.shape[1] + [0,0,0.05]
    ub = [1e+1,1,1e+2]*D_.shape[1] + [1,1e+1,5]
    

    bounds = np.array([lb,ub]).T
    result  = least_squares(Multi_Hill, pd_params, bounds=[lb,ub])#, method='COBYLA')   
    pd_params = result.x
    # result  = minimize(Multi_Hill, pd_params, bounds=bounds, method='COBYLA')     
    # pd_params = result.x
    # pd_params = np.minimum(ub,np.maximum(lb, pd_params))
    result  = least_squares(Multi_Hill, pd_params, bounds=[lb,ub])#, method='COBYLA')   
    pd_params = result.x
    
    return pd_params

def sim_patient(pd_params,D_,Timesteps):
    E = np.zeros((Timesteps,))
    for t in range(Timesteps):
        PD_val = 1 - (pd_params[-3] * np.exp( -1*((np.log(t+1) - pd_params[-2])**2) / ( 2*(pd_params[-1])**2 ) ))# / ((t+1)*pd_params[-1]*np.sqrt(2*np.pi))
        # PD_val = 1 - (pd_params[-3] * np.exp( -1*((t+1) - pd_params[-2])/ ( 2*(pd_params[-1])**2 ) ) )
        for i in range(0, D_.shape[1]) :
            num1 = D_[t,i]
            num2 = pd_params[3*i+1]
            den1 = pd_params[3*i]
            den2 = D_[t,i]
            PD_val = PD_val + ( num2 * ( num1**(pd_params[3*i + 2]) )/( den1**(pd_params[3*i + 2]) + den2**(pd_params[3*i + 2]) ) )
        PD_val = pd_params[-3] * PD_val
        PD_val = np.minimum(1,np.maximum(0, PD_val))
        E[t] = (1-PD_val)
    return E
        

  
            
