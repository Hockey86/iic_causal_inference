#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:43:45 2019

@author: harshparikh
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
warnings.filterwarnings("ignore")

class malts:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,k=10,reweight=False):
        # np.random.seed(0)
        self.C = C #coefficient to regularozation term
        self.k = k
        self.reweight = reweight
        self.n, self.p = data.shape
        self.p = self.p - 2 #shape of the data
        self.outcome = outcome
        self.treatment = treatment
        self.discrete = discrete
        self.continuous = list(set(data.columns).difference(set([outcome]+[treatment]+discrete)))
        
        self.df = data.copy(deep=True)
        self.Xc = self.df[self.continuous].to_numpy()
        self.Xd = self.df[self.discrete].to_numpy()
        self.Y  = self.df[self.outcome].to_numpy()
        self.T  = self.df[self.treatment].to_numpy()
        
        self.del2_Y = ((np.ones((len(self.Y),len(self.Y)))*self.Y).T - (np.ones((len(self.Y),len(self.Y)))*self.Y))**2
        self.del2_T = ((np.ones((len(self.T),len(self.T)))*self.T).T - (np.ones((len(self.T),len(self.T)))*self.T))**2
        self.Dc = np.ones((self.Xc.shape[0],self.Xc.shape[1],self.Xc.shape[0])) * self.Xc.T
        self.Dc = (self.Dc - self.Dc.T)
        self.Dd = np.ones((self.Xd.shape[0],self.Xd.shape[1],self.Xd.shape[0])) * self.Xd.T
        self.Dd = (self.Dd != self.Dd.T) 
        

    def threshold(self,x):
        k = self.k
        for i in range(x.shape[0]):
            row = x[i,:]
            row1 = np.where( row < row[np.argpartition(row,k+1)[k+1]],1,0)
            x[i,:] = row1
        return x
    
    def distance(self,Mc,Md,xc1,xd1,xc2,xd2):
        dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
        dd = np.sum((Md**2)*xd1!=xd2)
        return dc+dd
        
    def loss_(self, Mc, Md, xc1, xd1, y1, xc2, xd2, y2, gamma=1 ):
        w12 = np.exp( -1 * gamma * self.distance(Mc,Md,xc1,xd1,xc2,xd2) )
        return w12*((y1-y2)**2)
    
    def calcW(self,Mc,Md):
        #this step is slow
        Dc = np.sum( ( self.Dc * (Mc.reshape(-1,1)) )**2, axis=1)
        Dd = np.sum( ( self.Dd * (Md.reshape(-1,1)) )**2, axis=1)
        W = self.threshold( (Dc + Dd) )
        W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
        return W
    
    def Delta_(self,Mc,Md):
            self.W = self.calcW(Mc,Md)
            self.delta = np.sum((self.T - (np.matmul(self.W,self.T) - np.diag(self.W)*self.T))**2)
            return self.delta
    
    def objective(self,M):
        Mc = M[:len(self.continuous)]
        Md = M[len(self.continuous):]
        delta = self.Delta_(Mc,Md)
        reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 )
        cons1 = 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2
        cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md)) < 0 ) )
        return delta + reg + cons1 + cons2
        
    def fit(self,method='COBYLA'):
        # np.random.seed(0)
        M_init = np.ones((self.p,))
        res = opt.minimize( self.objective, x0=M_init,method=method )
        self.M = res.x
        self.Mc = self.M[:len(self.continuous)]
        self.Md = self.M[len(self.continuous):]
        self.M_opt = pd.DataFrame(self.M.reshape(1,-1),columns=self.continuous+self.discrete,index=['Diag'])
        return res
    
    def get_matched_groups(self, df_estimation, k=10 ):
        #units to be matched
        Xc = df_estimation[self.continuous].to_numpy()
        Xd = df_estimation[self.discrete].to_numpy()
        Y = df_estimation[self.outcome].to_numpy()
        T = df_estimation[self.treatment].values
        
        df_T = {}
        Xc_T = {}
        Xd_T = {}
        Y_T = {}
        D_T = {}
        Dc_T = {}
        Dd_T = {}
        D_T = {}
        #splitted estimation data for matching
        df = df_estimation
        Xc = df[self.continuous].to_numpy()
        Xd = df[self.discrete].to_numpy()
        D = np.zeros((Y.shape[0],Y.shape[0]))
            
        #distance_treated
        Dc = (np.ones((Xc.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T).T)
        Dc = np.sum( (Dc * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
            
        Dd = (np.ones((Xd.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T).T )
        Dd = np.sum( (Dd * (self.Md.reshape(-1,1)) )**2 , axis=1 )
        D = (Dc + Dd).T
            
        MG = {}
        index = df_estimation.index
        for i in range(Y.shape[0]):
            matched_df = pd.DataFrame(np.hstack((Xc[i], Xd[i], Y[i], 0, T[i])).reshape(1,-1), index=['query'], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment])
            #finding k closest units to unit i
            idx = np.argpartition(D[i,:],k)
            matched_df_ = pd.DataFrame( np.hstack( (Xc[idx[:k],:], Xd[idx[:k],:].reshape((k,len(self.discrete))), Y[idx[:k]].reshape(-1,1), D[i,idx[:k]].reshape(-1,1),  T[idx[:k]].reshape(-1,1)) ), index=df.index[idx[:k]], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] )
            matched_df = matched_df.append(matched_df_)
            MG[index[i]] = matched_df
            #{'unit':[ Xc[i], Xd[i], Y[i], T[i] ] ,'control':[ matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C],'treated':[matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T ]}
        MG_df = pd.concat(MG)
        return MG_df
    
    def CATE(self,MG,outcome_discrete=False,model='linear'):
        cate = {}
        for k in pd.unique(MG.index.get_level_values(0)):
            matched_df = MG.loc[k]
            #each treatment arm t_level
            matched_T_X = matched_df[[self.treatment]+self.continuous+self.discrete]
            matched_Y = matched_df[self.outcome]
            matched_m = lm.RidgeCV().fit(matched_T_X,matched_Y)
            diameter = matched_df['distance'].max()
            
            cate[k]['CATE'] = matched_m.coef_[0]
            cate[k]['diameter'] = diameter

            cate[k]['Y'] = matched_df.loc['query'][self.outcome] 
            cate[k]['T'] = matched_df.loc['query'][self.treatment]
                               
        return pd.DataFrame.from_dict(cate,orient='index')   
            
        
class malts_mf:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,k_tr=10,k_est=10,estimator='linear',smooth_cate=True,reweight=False,n_splits=5,n_repeats=1,output_format='brief'):
        self.n_splits = n_splits
        self.C = C
        self.k_tr = k_tr
        self.k_est = k_est
        self.outcome = outcome
        self.treatment = treatment
        self.treatment_levels = list(data[treatment].unique())
        self.discrete = discrete
        self.continuous = list(set(data.columns).difference(set([outcome]+[treatment]+discrete)))
        self.reweight = reweight
        
        skf = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
        gen_skf = skf.split(data)
        self.M_opt_list = []
        self.MG_list = []
        self.CATE_df = pd.DataFrame()
        N = np.zeros((data.shape[0],data.shape[0]))
        self.MG_matrix = pd.DataFrame(N, columns=data.index, index=data.index)
        
        i = 0
        for est_idx, train_idx in gen_skf:
            df_train = data.iloc[train_idx]
            df_est = data.iloc[est_idx]
            m = malts( outcome, treatment, treatment_levels = self.treatment_levels, data=df_train, discrete=discrete, C=self.C, k=self.k_tr, reweight=self.reweight )
            m.fit()
            self.M_opt_list.append(m.M_opt)
            mg = m.get_matched_groups(df_est,k_est)
            self.MG_list.append(mg)
            self.CATE_df = pd.concat([self.CATE_df, m.CATE(mg,model=estimator)], join='outer', axis=1)
        
        for i in range(n_splits*n_repeats):
            mg_i = self.MG_list[i]
            for a in mg_i.index:
                if a[1]!='query':
                    self.MG_matrix.loc[a[0],a[1]] = self.MG_matrix.loc[a[0],a[1]]+1
        
        cate_df = self.CATE_df[['CATE','diameter']]
        cate_df[self.outcome] = self.CATE_df['Y'].values[:,0]
        cate_df[self.treatment] = self.CATE_df['T'].values[:,0]
        self.CATE_df = cate_df