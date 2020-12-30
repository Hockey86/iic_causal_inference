#!/usr/bin/env python
# coding: utf-8
import copy
from collections import Counter, defaultdict
from itertools import combinations
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from mord import LogisticAT
from tqdm import tqdm


class MyCalibrator:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        
    def fit(self, X, y):
        yp = self.predict(X)
        self.recalibration_mapper = LogisticAT(alpha=0).fit(yp.reshape(-1,1), y)
        return self
    
    def predict(self, X):
        K = len(self.base_estimator.classes_)
        yp = np.sum(self.base_estimator.predict_proba(X)*np.arange(K), axis=1)
        return yp
        
    def predict_proba(self, X):
        yp = self.predict(X)
        yp2 = self.recalibration_mapper.predict_proba(yp.reshape(-1,1))
        return yp2


class MyLogisticRegression(LogisticRegression):
    """
    Allows bounds
    Removes regularization on intercept
    """
    def __init__(self, class_weight=None, tol=1e-6, C=1.0, l1_ratio=0., random_state=None, max_iter=1000, bounds=None):
        super().__init__(penalty='elasticnet', dual=False, tol=tol, C=C,
                 fit_intercept=True, intercept_scaling=1, class_weight=class_weight,
                 random_state=random_state, max_iter=max_iter,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=l1_ratio)
        self.bounds = bounds
                 
    def fit(self, X, y, sample_weight=None):
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        y = self.label_encoder.transform(y)
        
        def func(w, X, y, alpha, l1_ratio, sw):
            out, grad = _logistic_loss_and_grad(w, X, y, 0, sw)
            out_penalty = 0.5*alpha*(1 - l1_ratio)*np.sum(w[:-1]**2) + alpha*l1_ratio*np.sum(np.abs(w[:-1]))
            grad_penalty = np.r_[alpha*(1-l1_ratio)*w[:-1]+alpha*l1_ratio*np.sign(w[:-1]) ,0]
            return out+out_penalty, grad+grad_penalty
        
        y2 = np.array(y)
        y2[y2==0] = -1
        w0 = np.r_[np.random.randn(X.shape[1])/10, 0.]
        if self.bounds is None:
            method = 'BFGS'
        else:
            method = 'L-BFGS-B'
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight)
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))
        self.opt_res = minimize(
            func, w0, method=method, jac=True,
            args=(X, y2, 1./self.C, self.l1_ratio, sample_weight),
            bounds=self.bounds+[(None,None)],
            options={"gtol": self.tol, "maxiter": self.max_iter}
        )
        coef_ = self.opt_res.x[:-1]
        intercept_ = self.opt_res.x[-1]
        
        self.coef_ = coef_.reshape(1,-1)
        self.intercept_ = intercept_.reshape(1,)
        return self
        

class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    self.fit includes standardization and imputation.
    
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator, classes,
                    class_weight=None, min_level_diff=1,
                    binary_indicator=None, impute_KNN_K=10,
                    verbose=False):
        super().__init__()
        self.estimator = estimator
        self.classes = classes
        self.class_weight = class_weight
        self.min_level_diff = min_level_diff
        self.binary_indicator = binary_indicator
        self.impute_KNN_K = impute_KNN_K
        self.verbose = verbose
        
    #def __setattr__(self, name, value):
    #    setattr(self.estimator, name, value)
    #    super().__setattr__(name, value)
        
    def _generate_pairs(self, X, y, sample_weight):
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(range(len(X)), 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( max(sample_weight[i], sample_weight[j]) )
        
        if sample_weight is None:
            sw2 = None
        else:
            sw2 = np.array(sw2)

        return np.array(X2), np.array(y2), sw2

    def fit(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        
        # standardize
        if self.binary_indicator is None:
            self.binary_indicator = np.zeros(X.shape[1])==1
        self.Xmean = np.nanmean(X[:,~self.binary_indicator], axis=0)
        self.Xstd = np.nanstd(X[:,~self.binary_indicator], axis=0)
        self.Xmean[np.isnan(self.Xmean)] = 0
        self.Xstd[np.isnan(self.Xstd)|(self.Xstd==0)] = 1
        X[:,~self.binary_indicator] = (X[:,~self.binary_indicator]-self.Xmean)/self.Xstd
        
        # impute missing value
        self.imputer = KNNImputer(n_neighbors=self.impute_KNN_K).fit(X)
        if np.any(np.isnan(X)):
            X = self.imputer.transform(X)
            X[:,self.binary_indicator] = (X[:,self.binary_indicator]>0.5).astype(float)
            
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight, prior_count=2)
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))
        
        # generate pairs
        X2, y2, sw2 = self._generate_pairs(X, y, sample_weight)
        sw2 = sw2/sw2.mean()
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        self.estimator.fit(X2, y2, sample_weight=sw2)

        # get the mean of z for each level of y
        self.classes_ = self.classes
        z = self.predict_z(X, preprocess=False)
        for tol in range(0,len(self.classes_)//2):
            z_means = np.array([z[(y>=cl-tol)&(y<=cl+tol)].mean() for cl in self.classes_])
            if tol==0:
                self.z_means = z_means
            if np.all(np.diff(z_means)>0):
                self.z_means = z_means
                break

        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        return self

    def predict_z(self, X, preprocess=True):
        if preprocess:
            X = np.array(X)
            # standardize
            X[:,~self.binary_indicator] = (X[:,~self.binary_indicator]-self.Xmean)/self.Xstd
            # impute missing value
            if np.any(np.isnan(X)):
                X = self.imputer.transform(X)
                X[:,self.binary_indicator] = (X[:,self.binary_indicator]>0.5).astype(float)
        z = self.estimator.decision_function(X)
        return z

    def decision_function(self, X):
        z = self.predict_z(X)
        return z

    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        dists[np.isnan(dists)] = -np.inf
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        yp1d = self.predict_z(X)
        #yp = self.predict_proba(X)
        #yp1d = self.classes_[np.argmax(yp, axis=1)]
        return yp1d


def get_sample_weights(y, class_weight='balanced', prior_count=0):
    assert y.min()==0 ## assume y=[0,1,...]
    K = y.max()+1
    class_weights = {k:1./(np.sum(y==k)+prior_count) for k in range(K)}
    sw = np.array([class_weights[yy] for yy in y])
    sw = sw/np.mean(sw)
    return sw


def get_perf(model_type, y, yp, yp_prob):
    yp_int = np.argmax(yp_prob, axis=1)
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
        yptr = model.predict(Xtr)
        yptr_prob = model.predict_proba(Xtr)
        
        models.append(model)
        cv_tr_score.append(get_perf(model_type, ytr, yptr, yptr_prob))
        
        if len(cv_sids)>0:
            Xte = X[teids]
            yte = y[teids]
            sids_te = sids[teids]
            
            ypte_z = model.base_estimator.predict(Xte)
            ypte = model.predict(Xte)
            ypte_prob = model.predict_proba(Xte)
            
            df_te = pd.DataFrame(
                        data=np.c_[yte, ypte_z, ypte, ypte_prob],
                        columns=['y', 'z', 'yp']+['prob(%d)'%k for k in classes])
            df_te['SID'] = sids_te
            y_yp_te.append(df_te)
            cv_te_score.append(get_perf(model_type, yte, ypte, ypte_prob))
    
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
                exec(f'model.{pp[0]}.{pp[1]} = {val}')  # TODO assumes two
            else:
                exec(f'model.{p} = {val}')
        model.fit(X, y)
            
        # calibrate
        model = MyCalibrator(model)
        model.fit(X, y)
        models.append(model)
        
    return models, params, cv_tr_score, cv_te_score, y_yp_te
    


if __name__=='__main__':

    #data_type = 'humanIIC'
    data_type = 'CNNIIC'
    
    response_tostudy = 'iic_burden_smooth'
    #response_tostudy = 'spike_rate'

    Nbt = 1#000
    Ncv = 5
    model_type = 'ltr'
    n_jobs = 12
    random_state = 2020
    
    model = 'cauchy_expit_lognormal_drugoutside_ARMA2,6'
    maxiter = 1000
    
    with open(f'../data_to_fit_{data_type}_{response_tostudy}.pickle', 'rb') as f:
        res = pickle.load(f)
    for k in res:
        exec(f'{k} = res["{k}"]')
    pseudoMRNs = np.array(pseudoMRNs)
        
    df_simulator_params = pd.read_csv(f'../step6_simulator/results_{response_tostudy}/params_mean_{data_type}_{model}_iter{maxiter}.csv')
    # MAP = 1/3 SBP + 2/3 DBP
    # The sixth report of the Joint National Committee on prevention, detection, evaluation, and treatment of high blood pressure. [Arch Intern Med. 1997]
    simulator_param_names = list(df_simulator_params.columns)
    sids = df_simulator_params.SID.values
    df_simulator_params['mean arterial pressure'] = df_simulator_params['systolic BP']/3+df_simulator_params['diastolic BP']/3*2
    
    remove_names = [
        'SID', 'cluster',
        'iGCS = T?', 'iGCS-E', 'iGCS-V', 'iGCS-M', 'Worst GCS Intubation status', 'iGCS actual scores', 'APACHE II  first 24',
        'systolic BP', 'diastolic BP',
        'b[lacosamide]', 'b[midazolam]', 'b[pentobarbital]', 'b[phenobarbital]', 'b[valproate]']
    for x in remove_names:
        simulator_param_names.remove(x)

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
            np.percentile(D[i], 99, axis=0),
            np.mean(D[i], axis=0),
            pos_drug_mean,
            ])
    Dnames = ['max_dose_'+x for x in Dname] + ['mean_dose_'+x for x in Dname] + ['mean_positive_dose_'+x for x in Dname]
    
    # create X and y
    y = Y
    X = np.c_[np.array(D2), df_simulator_params[simulator_param_names].values]
    Xnames = Dnames + simulator_param_names
    
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
    cv_split_path = 'cv_split_Ncv%d_random_state%d.csv'%(Ncv, random_state)
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
    else:
        print('tr score:', tr_scores_bt[0])
        print('te score:', te_scores_bt[0])
    
    y_yps = []
    for bti, y_yp in enumerate(y_yp_bt):
        for cvi, y_yp_cv in enumerate(y_yp):
            N = len(y_yp_cv)
            cols = list(y_yp_cv.columns)
            y_yp_cv['bti'] = np.zeros(N)+bti
            y_yp_cv['cvi'] = np.zeros(N)+cvi
            y_yps.append(y_yp_cv[['bti', 'cvi']+cols])
    y_yps = pd.concat(y_yps, axis=0)
    y_yps.to_csv('cv_predictions_%s_Nbt%d.csv'%(model_type, Nbt), index=False)
    
    with open('results_%s_Nbt%d.pickle'%(model_type, Nbt), 'wb') as ff:
        pickle.dump({'tr_scores_bt':tr_scores_bt,
                     'te_scores_bt':te_scores_bt,
                     'coefs_bt':coefs_bt,
                     'y_yp_bt':y_yp_bt,
                     'params':params,
                     'model':final_model,
                     'Xnames':Xnames,
                     'Dmax':Dmax,
                    }, ff)
                    
