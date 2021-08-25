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
    def __init__(self,outcome,treatment,treatment_levels,data,discrete=[],C=1,k=1,reweight=False):
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
        
        self.treatment_levels = treatment_levels
        #splitting the data for each treatment arm
        self.df_T = {}
        self.Xc_T = {}
        self.Xd_T = {}
        self.Y_T = {}
        self.del2_Y_T = {}
        self.Dc_T = {}
        self.Dd_T = {}
        
        for t_level in self.treatment_levels:
            self.df_T[t_level] = data.loc[data[treatment]==t_level]
            self.Xc_T[t_level] = self.df_T[t_level][self.continuous].to_numpy()
            self.Xd_T[t_level] = self.df_T[t_level][self.discrete].to_numpy()
            self.Y_T[t_level]  = self.df_T[t_level][self.outcome].to_numpy()
            self.del2_Y_T[t_level] = ((np.ones((len(self.Y_T[t_level]),len(self.Y_T[t_level])))*self.Y_T[t_level]).T - (np.ones((len(self.Y_T[t_level]),len(self.Y_T[t_level])))*self.Y_T[t_level]))**2
            self.Dc_T[t_level] = np.ones((self.Xc_T[t_level].shape[0],self.Xc_T[t_level].shape[1],self.Xc_T[t_level].shape[0])) * self.Xc_T[t_level].T
            self.Dc_T[t_level] = (self.Dc_T[t_level] - self.Dc_T[t_level].T)
            self.Dd_T[t_level] = np.ones((self.Xd_T[t_level].shape[0],self.Xd_T[t_level].shape[1],self.Xd_T[t_level].shape[0])) * self.Xd_T[t_level].T
            self.Dd_T[t_level] = (self.Dd_T[t_level] != self.Dd_T[t_level].T) 
        

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
    
    def calcW_T(self,Mc,Md,t_level):
        #this step is slow
        Dc = np.sum( ( self.Dc_T[t_level] * (Mc.reshape(-1,1)) )**2, axis=1)
        Dd = np.sum( ( self.Dd_T[t_level] * (Md.reshape(-1,1)) )**2, axis=1)
        W = self.threshold( (Dc + Dd) )
        W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
        return W
    
    def Delta_(self,Mc,Md):
        self.W_T = {}
        self.delta_T = {}
        for t_level in self.treatment_levels:
            try:
                self.W_T[t_level] = self.calcW_T(Mc,Md,t_level)
                self.delta_T[t_level] = np.sum((self.Y_T[t_level] - (np.matmul(self.W_T[t_level],self.Y_T[t_level]) - np.diag(self.W_T[t_level])*self.Y_T[t_level]))**2)
            except:
                self.delta_T[t_level] = 0
        if self.reweight == False:
            s = 0
            for t_level in self.treatment_levels:
                s = s + self.delta_T[t_level]
                return s
        elif self.reweight == True:
            s = 0
            r = 0
            for t_level in self.treatment_levels:
                s = s + self.delta_T[t_level]/len(self.Y_T[t_level])
                r = r + len(self.Y_T[t_level])
                return r*s
    
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
    
    def get_matched_groups(self, df_estimation, k=1 ):
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
        for t_level in self.treatment_levels:
            df_T[t_level] = df_estimation.loc[df_estimation[self.treatment]==t_level]
            Xc_T[t_level] = df_T[t_level][self.continuous].to_numpy()
            Xd_T[t_level] = df_T[t_level][self.discrete].to_numpy()
            Y_T[t_level] = df_T[t_level][self.outcome].to_numpy()
            D_T[t_level] = np.zeros((Y.shape[0],Y_T[t_level].shape[0]))
            
        #distance_treated
            Dc_T[t_level] = (np.ones((Xc_T[t_level].shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_T[t_level].shape[0])) * Xc_T[t_level].T).T)
            Dc_T[t_level] = np.sum( (Dc_T[t_level] * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
            
            Dd_T[t_level] = (np.ones((Xd_T[t_level].shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_T[t_level].shape[0])) * Xd_T[t_level].T).T )
            Dd_T[t_level] = np.sum( (Dd_T[t_level] * (self.Md.reshape(-1,1)) )**2 , axis=1 )
            D_T[t_level] = (Dc_T[t_level] + Dd_T[t_level]).T
            
        MG = {}
        index = df_estimation.index
        for i in range(Y.shape[0]):
            matched_df = pd.DataFrame(np.hstack((Xc[i], Xd[i], Y[i], 0, T[i])).reshape(1,-1), index=['query'], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment])
            #finding k closest units to unit i with treatment arm t_level
            for t_level in self.treatment_levels:
                idx = np.argpartition(D_T[t_level][i,:],k)
                matched_df_t_level = pd.DataFrame( np.hstack( (Xc_T[t_level][idx[:k],:], Xd_T[t_level][idx[:k],:].reshape((k,len(self.discrete))), Y_T[t_level][idx[:k]].reshape(-1,1), D_T[t_level][i,idx[:k]].reshape(-1,1), np.array([t_level for ki in range(k)]).reshape(-1,1) ) ), index=df_T[t_level].index[idx[:k]], columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] )
                matched_df = matched_df.append(matched_df_t_level)
            MG[index[i]] = matched_df
            #{'unit':[ Xc[i], Xd[i], Y[i], T[i] ] ,'control':[ matched_Xc_C, matched_Xd_C, matched_Y_C, d_array_C],'treated':[matched_Xc_T, matched_Xd_T, matched_Y_T, d_array_T ]}
        MG_df = pd.concat(MG)
        return MG_df
    
    def CATE(self,MG,outcome_discrete=False,model='linear'):
        cate = {}
        for k in pd.unique(MG.index.get_level_values(0)):
            v = MG.loc[k]
            #each treatment arm t_level
            matched_X_T = {}
            matched_Y_T = {}
            x = v.loc['query'][self.continuous+self.discrete].to_numpy().reshape(1,-1)
            for t_level in self.treatment_levels:
                matched_Y_T[t_level] = v.loc[v[self.treatment]==t_level].drop(index='query',errors='ignore')[self.outcome]
            
            
            vc = v[self.continuous].to_numpy()
            vd = v[self.discrete].to_numpy()
            dvc = np.ones((vc.shape[0],vc.shape[1],vc.shape[0])) * vc.T.astype(float)
            dist_cont = np.sum( ( (dvc - dvc.T) * (self.Mc.reshape(-1,1)) )**2, axis=1) 
            dvd = np.ones((vd.shape[0],vd.shape[1],vd.shape[0])) * vd.T.astype(float)
            dist_dis = np.sum( ( (dvd - dvd.T) * (self.Md.reshape(-1,1)) )**2, axis=1) 
            dist_mat = pd.DataFrame(dist_cont + dist_dis, index = v.index, columns=v.index)
            diameter = { 'diameter(%s)'%(str(t_level)): np.max(dist_mat.loc[v[self.treatment]==t_level,'query']) for t_level in self.treatment_levels }
            
            cate[k] = { 'Y(%s)'%(str(t_level)): np.mean(matched_Y_T[t_level].astype(float)) for t_level in self.treatment_levels }
            cate[k].update(diameter)

            cate[k]['Y'] = v.loc['query'][self.outcome] 
            cate[k]['T'] = v.loc['query'][self.treatment]
                               
        return pd.DataFrame.from_dict(cate,orient='index')   

        
class malts_mf:
    def __init__(self,outcome,treatment,data,discrete=[],C=1,k_tr=2,k_est=2,estimator='linear',smooth_cate=True,reweight=False,n_splits=5,n_repeats=1,output_format='brief'):
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
        
        skf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
        gen_skf = skf.split(data,data[treatment])
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
        
        cate_df = self.CATE_df[['Y(%s)'%(str(t_level)) for t_level in self.treatment_levels]+['diameter(%s)'%(str(t_level)) for t_level in self.treatment_levels]]
        cate_df[self.outcome] = self.CATE_df['Y'].values[:,0]
        cate_df[self.treatment] = self.CATE_df['T'].values[:,0]
        self.CATE_df = cate_df