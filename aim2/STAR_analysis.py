import numpy as np
import pandas as pd
from scipy import special
from copy import copy
import math
import seaborn as sns
import importlib
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm
from tqdm import tqdm
warnings.filterwarnings("ignore")

import sklearn.linear_model as lm

import ame_permutation_test as pt
import sensitivity_analysis as sa
import gcm 
import dml_test
importlib.reload(gcm)
importlib.reload(sa)
importlib.reload(pt)
importlib.reload(dml_test)

sns.set(font_scale=2)

STAR_High_School = pd.read_spss('PROJECTSTAR/STAR_High_Schools.sav')
STAR_K3_School = pd.read_spss('PROJECTSTAR/STAR_K-3_Schools.sav').set_index('schid')
STAR_Students = pd.read_spss('PROJECTSTAR/STAR_Students.sav').set_index('stdntid')
Comparison_Students = pd.read_spss('PROJECTSTAR/Comparison_Students.sav').set_index('stdntid')

# pre-treatment covariates
gk_cols = list(filter(lambda x: 'gk' in x, STAR_Students.columns))
g1_cols = list(filter(lambda x: 'g1' in x, STAR_Students.columns))
g2_cols = list(filter(lambda x: 'g2' in x, STAR_Students.columns))
g3_cols = list(filter(lambda x: 'g3' in x, STAR_Students.columns))
g_cols = gk_cols+g1_cols+g2_cols+g3_cols

personal_cols = ['gender','race','birthmonth','birthday','birthyear']

cols_cond = ['surban',
            'tgen',
            'trace',
            'thighdegree',
            'tcareer',
            'tyears',
            'classsize',
            'freelunch']

class_sizes = ['g1classsize',
             'g2classsize']

g3scores = ['g3treadss',
            'g3tmathss',
            'g3tlangss',
            'g3socialsciss']

map(lambda x: x in s,cols_cond)
g_cols_cond = list(filter(lambda s: np.sum(list(map(lambda x: x in s,cols_cond)))>0,g_cols))
df_exp = STAR_Students[personal_cols]#+class_sizes]
df_exp['Sample'] = 1
df_exp['g3avgscore'] = STAR_Students[g3scores].mean(axis=1)
df_exp['g3smallclass'] = (STAR_Students['g3classsize']<=17).astype(int)

df_obs = Comparison_Students[personal_cols]#+class_sizes]
df_obs['Sample'] = 0
df_obs['g3avgscore'] = Comparison_Students[g3scores].mean(axis=1)
df_obs['g3smallclass'] = (Comparison_Students['g3classsize']<=17).astype(int)

df = df_exp.append(df_obs)
df_no_na = df.dropna()

df_no_na['birthmonth'].replace({'JANUARY':1.0,
                               'FEBRUARY':2.0,
                               'MARCH':3.0,
                               'APRIL':4.0,
                                'ARPIL':4.0,
                               'MAY':5.0,
                               'JUNE':6.0,
                               'JULY':7.0,
                               'AUGUST':8.0,
                               'SEPTEMBER':9.0,
                               'OCTOBER':10.0,
                               'NOVEMBER':11.0,
                                'DECEMBER':12.0},inplace=True)
df_no_na_dummified = pd.get_dummies(df_no_na,columns=['gender','race'],drop_first=True)

# DML TEST
N = df_no_na_dummified.shape[0]
df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df = dml_test.fit('g3avgscore','g3smallclass', 
                                                            df_exp=df_no_na_dummified.loc[df_no_na_dummified['Sample']==1].drop(columns=['Sample']), 
                                                            df_obs=df_no_na_dummified.loc[df_no_na_dummified['Sample']==0].drop(columns=['Sample']),
                                                            n_splits=N-1)
psi = dml_test.Psi(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, 'g3avgscore', 'g3smallclass')
p_val = dml_test.dml_pval(psi)

fig = plt.figure(figsize=(10,5))
N = psi.shape[0]
thetas = []
for repeat in range(10000):
    idx = list(np.random.randint(0,N,size=N))
    psi_bs = psi[idx,:]
    thetas += [np.nanmean(psi_bs, axis = 0)]
thetas = np.array(thetas)
sns.kdeplot(thetas[:,0])
sns.kdeplot(thetas[:,1])
plt.legend([r'$\theta(1)$',r'$\theta(0)$'])
plt.title('p(1) = %.6f, p(0) = %.6f'%(p_val[0],p_val[1]))
fig.savefig('psi_STAR.png')

# SENSITIVITY ANALYSIS POST-DML TEST
propensity_model = lm.LogisticRegressionCV().fit(df_no_na_dummified.drop(columns=['g3smallclass','g3avgscore']).loc[df_no_na_dummified['Sample']==0],
                                                 df_no_na_dummified.loc[df_no_na_dummified['Sample']==0]['g3smallclass'])
pi = propensity_model.predict_proba(df_no_na_dummified.drop(columns=['g3smallclass','g3avgscore']).loc[df_no_na_dummified['Sample']==0])
w = df_no_na_dummified.loc[df_no_na_dummified['Sample']==0]['g3smallclass'] * pi[:,0] + (1-df_no_na_dummified.loc[df_no_na_dummified['Sample']==0]['g3smallclass']) * pi[:,1]

def q(a,T):
    return a*(2*T - 1)

p_array = []
for a in tqdm(np.linspace(0,50,num=21)):
    df_no_na_dummified_copy = df_no_na_dummified.copy(deep=True)
    df_no_na_dummified_copy.loc[df_no_na_dummified_copy['Sample']==0,'g3avgscore'] = df_no_na_dummified_copy.loc[df_no_na_dummified_copy['Sample']==0]['g3avgscore'] - q(a,df_no_na_dummified_copy.loc[df_no_na_dummified_copy['Sample']==0]['g3smallclass']) * w
    df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df = dml_test.fit('g3avgscore','g3smallclass', 
                                                            df_exp=df_no_na_dummified_copy.loc[df_no_na_dummified_copy['Sample']==1].drop(columns=['Sample']), 
                                                            df_obs=df_no_na_dummified_copy.loc[df_no_na_dummified_copy['Sample']==0].drop(columns=['Sample']),
                                                            n_splits=N-1)
    psi = dml_test.Psi(df_mu, df_nu, df_p, df_pi_exp, df_pi_obs, df, 'g3avgscore', 'g3smallclass')
    p_val = dml_test.dml_pval(psi)
    p_array.append( p_val )
    

sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,50,num=21)/400,np.array(p_array)[:,0],lw=2.5)
plt.axhline(0.05,c='red',ls='--')
plt.ylabel('p(1)')
plt.xlabel(r'$\alpha$')
plt.title(r'selection bias: $ q(X,T) = \alpha(2T - 1)$')
plt.savefig('sensitivity_test_STAR.png')


    

