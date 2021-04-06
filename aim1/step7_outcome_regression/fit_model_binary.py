#!/usr/bin/env python
# coding: utf-8
import copy
import sys
from collections import Counter, defaultdict
from itertools import combinations
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score, cohen_kappa_score, f1_score, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
from myclasses import *
from fit_model_ordinal import generate_outcome_X, stratified_group_k_fold


def get_perf(model_type, y, yp_int, yp):
    ids = np.all(~np.isnan(np.c_[y, yp_int, yp]), axis=1)
    y = y[ids]
    yp_int = yp_int[ids]
    yp = yp[ids]
    #if model_type in ['lowess-logreg']:
    perf = pd.Series(
        data=[
            np.mean(y==yp_int),
            roc_auc_score(y,yp),
            cohen_kappa_score(y, yp_int),
            f1_score(y,yp_int, average='weighted'),
            balanced_accuracy_score(y,yp_int),
        ],
        index=[
            'accuracy',
            'AUC',
            'Cohen kappa',
            'weighted F1 score',
            'balanced accuracy',
        ])
    #else:
    #    raise ValueError('Unknown model_type:', model_type)
    return perf


def get_coef(model_type, model):
    if model_type=='logreg':
        coef = model.base_estimator.coef_.flatten()
    else:
        coef = None
    return coef
        
        
def fit_model(X, y, sids, cv_split_, binary_indicator, bounds, model_type='logreg', refit=True, best_params=None, n_jobs=1, random_state=None):
    models = []
    cv_tr_score = []
    cv_te_score = []
    y_yp_te = []
    
    if best_params is None:
        params = []
    else:
        params = best_params
        
    classes = np.arange(y.max()+1)
    n_classes = len(set(classes))
    cv_split = copy.deepcopy(cv_split_)
    Ncv = len(cv_split)
        
    # outer CV
    for cvi, cv_sids in enumerate(cv_split):#tqdm
        teids = np.in1d(sids, cv_sids)
        trids = ~teids
        Xtr = X[trids]
        ytr = y[trids]
        if len(set(ytr))!=len(set(y)):
            continue
            
        if model_type=='logreg':
            model_params = {'estimator__C':np.logspace(-1,1,3),
                            'estimator__l1_ratio':np.arange(0.5,0.8,0.1),#1
                            'impute_KNN_K':[5,10],#,50],
                           }
            metric = make_scorer(lambda y,yp:spearmanr(y,yp).correlation)
            model = LTRPairwise(MyLogisticRegression(
                                class_weight=None,
                                random_state=random_state,
                                max_iter=1000,
                                bounds=bounds,),
                            classes, class_weight='balanced', min_level_diff=2,
                            binary_indicator=binary_indicator,
                            verbose=False)
        elif model_type=='lowess-logreg':
            model_params = {'n_neighbors':[20,50,100],
                            'sigma':[0.5,0.6,0.7],
                            'impute_KNN_K':[5,10,50],
                            }
            metric = 'f1_weighted'
            model = LogisticRegression(
                    C=10., max_iter=1000,
                    penalty='l2', class_weight='balanced',
                    random_state=random_state)
            model = LOWESS(model, relative=False)
        elif model_type=='rf':
            model_params = {'n_estimators':[500,1000],
                            'max_depth':[2,3,5],
                            'min_samples_leaf':[10,20],
                            'ccp_alpha':[0.0001, 0.001,0.01],
                            'impute_KNN_K':[5,10,50],
                            }
            metric = 'f1_weighted'
            model = MyRandomForestClassifier(n_jobs=1, random_state=random_state, class_weight='balanced')
        elif model_type=='gbt':
            model_params = {'max_depth':[3,5,10],
                            'l2_regularization':[0.01,0.1,1],
                            }
            metric = 'f1_weighted'
            bound2cst = {(None,None):0, (None,0):-1, (0,None):1}
            monotonic_cst = [0 if binary_indicator[bi] else bound2cst[bounds[bi]] for bi in range(len(bounds))]
            model = HistGradientBoostingClassifier(
                    max_iter=1000, categorical_features=binary_indicator,
                    monotonic_cst=monotonic_cst, early_stopping=True,
                    random_state=random_state)
        else:
            raise ValueError('Unknown model_type:', model_type)
                        
        if best_params is None:
            model.n_jobs = 1
            model = GridSearchCV(model, model_params,
                        n_jobs=n_jobs, refit=True,
                        cv=Ncv, scoring=metric,
                        verbose=False)
        else:
            for p in params[cvi]:
                val = params[cvi][p]
                if '__' in p:
                    pp = p.split('__')
                    exec(f'model.{pp[0]}.{pp[1]} = {val}')  # TODO assumes two levels
                else:
                    exec(f'model.{p} = {val}')
            model.n_jobs = n_jobs
        model.fit(Xtr, ytr)
        
        if best_params is None and hasattr(model, 'best_params_'):
            params.append({p:model.best_params_[p] for p in model_params})
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        # calibrate
        model = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
        model.fit(Xtr, ytr)
        
        yptr_int = model.predict(Xtr)
        yptr_prob = model.predict_proba(Xtr)[:,1]
        
        models.append(model)
        cv_tr_score.append(get_perf(model_type, ytr, yptr_int, yptr_prob))
        
        if len(cv_sids)>0:
            Xte = X[teids]
            yte = y[teids]
            sids_te = sids[teids]
            
            ypte_int = model.predict(Xte)
            ypte_prob = model.predict_proba(Xte)[:,1]
            
            df_te = pd.DataFrame(
                        data=np.c_[yte, ypte_int, ypte_prob],
                        columns=['y', 'yp_int', 'yp_prob'])
            df_te['SID'] = sids_te
            y_yp_te.append(df_te)
            cv_te_score.append(get_perf(model_type, yte, ypte_int, ypte_prob))
    
    cv_tr_score = sum(cv_tr_score)/len(cv_tr_score)
    cv_te_score = sum(cv_te_score)/len(cv_te_score)
    
    if refit:
        if model_type=='logreg':
            model = LTRPairwise(MyLogisticRegression(
                                class_weight=None,
                                random_state=random_state,
                                max_iter=1000,
                                bounds=bounds,),
                            classes, class_weight='balanced', min_level_diff=2,
                            binary_indicator=binary_indicator,
                            verbose=False)
        elif model_type=='lowess-logreg':
            model = LogisticRegression(
                    C=10., max_iter=1000,
                    penalty='l2', class_weight='balanced',
                    random_state=random_state)
            model = LOWESS(model, relative=False)
        elif model_type=='rf':
            model = MyRandomForestClassifier(n_jobs=n_jobs, random_state=random_state, class_weight='balanced')
        elif model_type=='gbt':
            model = HistGradientBoostingClassifier(
                    max_iter=1000, categorical_features=binary_indicator,
                    monotonic_cst=monotonic_cst, early_stopping=True,
                    random_state=random_state)
        else:
            raise ValueError('Unknown model_type:', model_type)
                        
        for p in params[0]:
            val = Counter([params[cvi][p] for cvi in range(Ncv)]).most_common()[0][0]
            if '__' in p:
                pp = p.split('__')
                exec(f'model.{pp[0]}.{pp[1]} = {val}')  # TODO assumes two levels
            else:
                exec(f'model.{p} = {val}')
            print(p, val)
        model.fit(X, y)
            
        # calibrate
        model = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
        model.fit(X, y)
        models.append(model)
        
    return models, params, cv_tr_score, cv_te_score, y_yp_te
    
    
if __name__=='__main__':

    input_type = str(sys.argv[1])#simulator_param or response
    assert input_type in ['simulator_param', 'response']
    
    #data_type = 'humanIIC'
    data_type = 'CNNIIC'
    
    responses = ['iic_burden_smooth', 'spike_rate']
    responses_txt = '+'.join(responses)
    
    Nbt = 0
    Ncv = 5
    model_type = str(sys.argv[2])
    simulator_type = 'cauchy_expit_lognormal_drugoutside_ARMA2,6'
    n_jobs = 12
    random_state = 2020
    
    with open(f'../data_to_fit_{data_type}_{responses_txt}.pickle','rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec(f'{k} = res[\'{k}\']')
    y = Y.astype(int)
    
    # standardize drugs
    Dmax = []
    for di in range(D[0].shape[-1]):
        dd = np.concatenate([x[:,di] for x in D])
        dd[dd==0] = np.nan
        Dmax.append(np.nanpercentile(dd,95))
    Dmax = np.array(Dmax)
    #for i in range(len(D)):
    #    D[i] = D[i]/Dmax
    X, Xnames = generate_outcome_X(Pobs, D, Dmax, Dname, input_type, responses, W)
    X = np.c_[X, C]
    Xnames.extend(Cname)
    binary_indicator = np.array([set(X[:,i][~np.isnan(X[:,i])])=={0,1} for i in range(X.shape[1])])
    
    # convert to binary
    y[y<=3] = 0
    y[y>=4] = 1
    
    pos_Xnames = [
        'Age',
        'Hx CVA (including TIA)',
        'Hx HTN',
        'Hx Sz /epilepsy',
        'Hx brain surgery',
        'Hx CKD',
        'Hx CAD/MI',
        'Hx CHF',
        'Hx DM',
        'Hx of HLD',
        'Hx tobacco (including ex-smokers)',
        'Hx ETOH abuse any time in their life (just when in the hx is mentioned)',
        'Hx other substance abuse, any time in their life',
        'Hx cancer (other than CNS cancer)',
        'Hx CNS cancer',
        'Hx COPD/ Asthma',
        'premorbid MRS before admission  (modified ranking scale),before admission',
        'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)',
        'hydrocephalus  (either on admission or during hospital course)   QPID',
        'iMV  (initial (on admission) mechanical ventilation)',
        'Midline shift with any reason ( Document Date)',
        'Primary systemic dx Sepsis/Shock',
        'neuro_dx_Seizures/status epilepticus',
        'prim_dx_Respiratory disorders',
        ]
    neg_Xnames = [
        'iGCS-Total',
        #'Worst GCS in 1st 24',
        ]
    bounds = []
    for xn in Xnames:
        if xn in pos_Xnames or xn.startswith('burden_'):  # this includes drugs and IIC and spike rate
            bounds.append((0,None))
        elif xn in neg_Xnames:
            bounds.append((None,0))
        else:
            bounds.append((None, None))
    
    # generate CV split
    cv_split_path = f'cv_split_Ncv{Ncv}_random_state{random_state}_{responses_txt}.csv'
    if not os.path.exists(cv_split_path):
        cv_split = np.zeros(len(X))
        for cvi, (_, teid) in enumerate(stratified_group_k_fold(X, y, pseudoMRNs, Ncv, seed=random_state)):
            cv_split[teid] = cvi
        pd.DataFrame(data={'SID':sids, 'PseudoMRN':pseudoMRNs, 'y':y, 'CV':cv_split}).to_csv(cv_split_path, index=False)
    df_cv = pd.read_csv(cv_split_path)
    assert [len(set(df_cv.PseudoMRN[df_cv.CV==k])&set(df_cv.PseudoMRN[df_cv.CV!=k])) for k in range(Ncv)] == [0]*Ncv
    assert [len(set(df_cv.SID[df_cv.CV==k])&set(df_cv.SID[df_cv.CV!=k])) for k in range(Ncv)] == [0]*Ncv
    cv_split = [df_cv.SID[df_cv.CV==i].values for i in range(Ncv)]
    
    # fit model with bootstrap
    tr_scores_bt = []
    te_scores_bt = []
    y_yp_bt = []
    coefs_bt = []
    params = None
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            ybt = y
            Xbt = X
            sidsbt = sids
        else:
            btids = np.random.choice(len(X), len(X), replace=True)
            ybt = y[btids]
            if len(set(ybt))!=len(set(y)):
                continue
            Xbt = X[btids]
            sidsbt = sids[btids]
        
        models, params, cv_tr_score, cv_te_score, y_yp = fit_model(
                            Xbt, ybt, sidsbt, cv_split,
                            binary_indicator, bounds,
                            model_type, refit=True,#bti==0,
                            best_params=params, n_jobs=n_jobs,
                            random_state=random_state+bti)
        tr_scores_bt.append(cv_tr_score)
        te_scores_bt.append(cv_te_score)
        y_yp_bt.append(y_yp)
        coefs_bt.append(get_coef(model_type, models[-1]))
        
        if bti==0:
            final_model = models[-1]
            print('tr score', cv_tr_score)
            print('te score', cv_te_score)

    if Nbt>0:
        for idx in tr_scores_bt[0].index:
            print(idx)
            print('tr score: %f [%f -- %f]'%(
                tr_scores_bt[0][idx],
                np.percentile([x[idx] for x in tr_scores_bt[1:]], 2.5),
                np.percentile([x[idx] for x in tr_scores_bt[1:]], 97.5),))
            print('te score: %f [%f -- %f]'%(
                te_scores_bt[0][idx],
                np.percentile([x[idx] for x in te_scores_bt[1:]], 2.5),
                np.percentile([x[idx] for x in te_scores_bt[1:]], 97.5),))
    
    if model_type=='logreg':
        df_coef = pd.DataFrame(data={'coef':coefs_bt[0], 'name':Xnames})
        df_coef = df_coef[['name', 'coef']]
        df_coef = df_coef.sort_values('coef', ascending=False).reset_index(drop=True)
        df_coef.to_csv(f'coef_{model_type}_{responses_txt}_{input_type}.csv', index=False)
    
    import pdb;pdb.set_trace()
    y_yps = []
    for bti, y_yp in enumerate(y_yp_bt):
        for cvi, y_yp_cv in enumerate(y_yp):
            N = len(y_yp_cv)
            cols = list(y_yp_cv.columns)
            y_yp_cv['bti'] = np.zeros(N)+bti
            y_yp_cv['cvi'] = np.zeros(N)+cvi
            y_yps.append(y_yp_cv[['bti', 'cvi']+cols])
    y_yps = pd.concat(y_yps, axis=0)
    y_yps.to_csv(f'cv_predictions_{model_type}_Nbt{Nbt}_{responses_txt}_{input_type}.csv', index=False)
    
    res = {'tr_scores_bt':tr_scores_bt,
             'te_scores_bt':te_scores_bt,
             'y_yp_bt':y_yp_bt,
             'params':params,
             'model':final_model,
             'Xnames':Xnames,
             'Dmax':Dmax,}
    if model_type=='logreg':
        res['coefs_bt'] = coefs_bt
    with open(f'results_{model_type}_Nbt{Nbt}_{responses_txt}_{input_type}.pickle', 'wb') as ff:
        pickle.dump(res, ff)

