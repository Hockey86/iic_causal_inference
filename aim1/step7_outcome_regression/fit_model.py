#!/usr/bin/env python
# coding: utf-8
import copy
import sys
from collections import Counter, defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from tqdm import tqdm
from myclasses import MyCalibrator, MyLogisticRegression, LTRPairwise


def get_sample_weights(y, class_weight='balanced', prior_count=0):
    assert y.min()==0 ## assume y=[0,1,...]
    K = y.max()+1
    class_weights = {k:1./(np.sum(y==k)+prior_count) for k in range(K)}
    sw = np.array([class_weights[yy] for yy in y])
    sw = sw/np.mean(sw)
    return sw


def get_perf(model_type, y, yp_int, yp, yp_prob):
    if model_type=='ltr':
        perf = pd.Series(
            data=[
                spearmanr(y, yp).correlation,
                np.mean(np.abs(y-yp_int)<=0),
                np.mean(np.abs(y-yp_int)<=1),
                np.mean(np.abs(y-yp_int)<=2),
                np.mean(np.abs(y-yp_int)<=3),
            ],
            index=[
                'Spearman\'s R',
                'accuracy(0)',
                'accuracy(1)',
                'accuracy(2)',
                'accuracy(3)',
            ])
    else:
        raise ValueError('Unknown model_type:', model_type)
    return perf


def get_coef(model_type, model):
    if model_type=='ltr':
        coef = model.base_estimator.estimator.coef_.flatten()
    else:
        raise ValueError('Unknown model_type:', model_type)
    return coef
    
    
def stratified_group_k_fold(X, y, groups, K, seed=None):
    """
    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(K)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    np.random.seed(seed)
    np.random.shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(K):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(K):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
        
        
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
            
        if model_type=='ltr':
            model_params = {'estimator__C':np.logspace(-3,1,5),
                            'estimator__l1_ratio':np.arange(0.5,1,0.1),
                            'impute_KNN_K':[5,10,50],
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
        model = MyCalibrator(model)
        model.fit(Xtr, ytr)
        
        yptr_z = model.base_estimator.predict(Xtr)
        yptr_int = np.argmax(model.base_estimator.predict_proba(Xtr), axis=1)
        yptr = model.predict(Xtr)
        yptr_prob = model.predict_proba(Xtr)
        
        models.append(model)
        cv_tr_score.append(get_perf(model_type, ytr, yptr_int, yptr, yptr_prob))
        
        if len(cv_sids)>0:
            Xte = X[teids]
            yte = y[teids]
            sids_te = sids[teids]
            
            ypte_z = model.base_estimator.predict(Xte)
            ypte_int = np.argmax(model.base_estimator.predict_proba(Xte), axis=1)
            ypte = model.predict(Xte)
            ypte_prob = model.predict_proba(Xte)
            
            df_te = pd.DataFrame(
                        data=np.c_[yte, ypte_z, ypte_int, ypte, ypte_prob],
                        columns=['y', 'z', 'yp_int', 'yp']+['prob(%d)'%k for k in classes])
            df_te['SID'] = sids_te
            y_yp_te.append(df_te)
            cv_te_score.append(get_perf(model_type, yte, ypte_int, ypte, ypte_prob))
    
    cv_tr_score = sum(cv_tr_score)/len(cv_tr_score)
    cv_te_score = sum(cv_te_score)/len(cv_te_score)
    
    if refit:
        if model_type=='ltr':
            model = LTRPairwise(MyLogisticRegression(
                                class_weight=None,
                                random_state=random_state,
                                max_iter=1000,
                                bounds=bounds,),
                            classes, class_weight='balanced', min_level_diff=2,
                            binary_indicator=binary_indicator,
                            verbose=False)
        else:
            raise ValueError('Unknown model_type:', model_type)
                        
        for p in params[0]:
            val = Counter([params[cvi][p] for cvi in range(Ncv)]).most_common()[0][0]
            if '__' in p:
                pp = p.split('__')
                exec(f'model.{pp[0]}.{pp[1]} = {val}')  # TODO assumes two levels
            else:
                exec(f'model.{p} = {val}')
        model.fit(X, y)
            
        # calibrate
        model = MyCalibrator(model)
        model.fit(X, y)
        models.append(model)
        
    return models, params, cv_tr_score, cv_te_score, y_yp_te
    


if __name__=='__main__':

    input_type = str(sys.argv[1])#simulator_param or response
    assert input_type in ['simulator_param', 'response']
    
    #data_type = 'humanIIC'
    data_type = 'CNNIIC'
    
    responses = ['iic_burden_smooth', 'spike_rate']
    responses_txt = '_'.join(responses)
    
    Nbt = 0
    Ncv = 5
    model_type = 'ltr'
    n_jobs = 12
    random_state = 2020
    
    model = 'cauchy_expit_lognormal_drugoutside_ARMA2,6'
    maxiter = 1000
    
    res = {}
    sids = set()  # common_sids
    for response in responses:
        with open(f'../data_to_fit_{data_type}_{response}.pickle', 'rb') as f:
            res[response] = pickle.load(f)
        if len(sids)==0:
            sids.update(res[response]['sids'])
        else:
            sids &= set(res[response]['sids'])
    sids = np.array(sorted(sids, key=lambda x:int(x[len('sid'):])))
    
    Pobs = {}
    for k in ['W', 'Dname', 'Cname', 'Yname']:
        exec(f'{k} = res[responses[0]]["{k}"]')
    for ri, response in enumerate(responses):
        ids = [res[response]['sids'].index(sid) for sid in sids]
        for k in res[response]:
            if ri==0 and k in ['window_start_ids', 'D']:
                res2 = [res[response][k][x] for x in ids]
                exec(f'{k} = res2')
            elif ri==0 and k in ['cluster', 'pseudoMRNs', 'C', 'Y']:
                exec(f'{k} = np.array(res[response]["{k}"])[ids]')
            elif k=='Pobs':
                Pobs[response] = [res[response][k][x] for x in ids]
    
    # MAP = 1/3 SBP + 2/3 DBP
    # The sixth report of the Joint National Committee on prevention, detection, evaluation, and treatment of high blood pressure. [Arch Intern Med. 1997]
    MAP = C[:,Cname.index('systolic BP')]/3+C[:,Cname.index('diastolic BP')]/3*2
    C = np.c_[C, MAP]
    Cname.append('mean arterial pressure')
    
    remove_names = [
        #'SID', 'cluster',
        'iGCS = T?', 'iGCS-E', 'iGCS-V', 'iGCS-M', 'Worst GCS Intubation status', 'iGCS actual scores', 'APACHE II  first 24',
        'systolic BP', 'diastolic BP',]
    C = C[:,~np.in1d(Cname, remove_names)]
    for x in remove_names:
        Cname.remove(x)
    
    if input_type=='simulator_param':
        all_sim_names = ['alpha0', 'alpha[1]', 'alpha[2]', 'theta[1]',
           'theta[2]', 'theta[3]', 'theta[4]', 'theta[5]', 'theta[6]', 'sigma_err',
           #'b[lacosamide]',
           'b[levetiracetam]',
           #'b[midazolam]',
           #'b[pentobarbital]',
           #'b[phenobarbital]',
           'b[propofol]',
           #'b[valproate]'
           ]
        Xsim = []
        sim_name = []
        for response in responses:
            df_sim = pd.read_csv(f'../step6_simulator/results_{response}/params_mean_{data_type}_{response}_{model}_iter{maxiter}.csv')
            ids = [np.where(df_sim==sid)[0][0] for sid in sids]
            Xsim.append( df_sim[all_sim_names].iloc[ids].values )
            sim_name.extend([x+'_'+response for x in all_sim_names])
        Xsim = np.concatenate(Xsim, axis=1)
        
    elif input_type=='response':
        Xsim = np.zeros((len(sids), len(responses)))
        sim_name = [x+'_mean' for x in responses]
        for ri, response in enumerate(responses):
            for i in range(len(Pobs[response])):
                meanP = np.nanmean(Pobs[response][i])
                if np.isnan(meanP):
                    meanP = 0
                Xsim[i,ri] = meanP

    # standardize drugs
    Dmax = []
    for di in range(D[0].shape[-1]):
        dd = np.concatenate([x[:,di] for x in D])
        dd[dd==0] = np.nan
        Dmax.append(np.nanpercentile(dd,95))
    Dmax = np.array(Dmax)
    for i in range(len(D)):
        D[i] = D[i]/Dmax
        
    D2 = []
    for i in range(len(D)):
        Di = np.array(D[i])
        Di[Di<1e-6] = np.nan
        pos_drug_mean = np.nanmean(Di, axis=0)
        pos_drug_mean[np.isnan(pos_drug_mean)] = 0
        D2.append(np.r_[
            #np.percentile(D[i], 99, axis=0),
            #np.mean(D[i], axis=0),
            pos_drug_mean,
            ])
    #Dname = ['max_dose_'+x for x in Dname] + \
    #        ['mean_dose_'+x for x in Dname] + \
    Dname = ['mean_positive_dose_'+x for x in Dname]
    
    # create X and y
    y = Y
    X = np.c_[np.array(D2), Xsim, C]
    Xnames = Dname + sim_name + Cname
    
    ids = ~np.isnan(y)
    y = y[ids].astype(int)
    X = X[ids]
    sids = sids[ids]
    pseudoMRNs = pseudoMRNs[ids]
    
    binary_indicator = np.array([set(X[:,i])=={0,1} for i in range(X.shape[1])])
    
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
        'Worst GCS in 1st 24',
        ]
    bounds = []
    for xn in Xnames:
        if xn in pos_Xnames:
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
    
    df_coef = pd.DataFrame(data={'coef':coefs_bt[0], 'name':Xnames})
    df_coef = df_coef[['name', 'coef']]
    df_coef = df_coef.sort_values('coef', ascending=False).reset_index(drop=True)
    df_coef.to_csv(f'coef_{model_type}_{responses_txt}_{input_type}.csv', index=False)
    
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
    
    with open(f'results_{model_type}_Nbt{Nbt}_{responses_txt}_{input_type}.pickle', 'wb') as ff:
        pickle.dump({'tr_scores_bt':tr_scores_bt,
                     'te_scores_bt':te_scores_bt,
                     'coefs_bt':coefs_bt,
                     'y_yp_bt':y_yp_bt,
                     'params':params,
                     'model':final_model,
                     'Xnames':Xnames,
                     'Dmax':Dmax,
                    }, ff)
                    
