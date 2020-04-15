from collections import Counter, defaultdict
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


def stratified_group_k_fold(X, y, groups, k, seed=None):
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
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def myfit(X, y, sids, ncv=10, random_state=None):
    
    Cs = []
    cv_ids = []
    ytes = []
    yptes = []
    sids_tes = []
    
    #split data in train set and test set
    for k, (train, test) in enumerate(stratified_group_k_fold(X, y, sids, k=ncv, seed=random_state)):
        Xtr = X[train]
        Xte = X[test] 
        ytr = y[train]
        yte = y[test]
        sids_te = sids[test]
        
        # TODO feature selection
        
        # inner CV loop
        model = Pipeline(steps=(
                    ('standardizer', StandardScaler()),
                    ('model', LogisticRegression(penalty='l1',
                                class_weight='balanced',
                                random_state=random_state,
                                solver='saga', max_iter=10000,
                                l1_ratio=None))))
        model = GridSearchCV(model, {'model__C':[0.01, 0.1, 1, 10,100]},
                            scoring='f1_weighted', n_jobs=1, cv=ncv)
        model.fit(Xtr, ytr)
        
        Cs.append(model.best_params_['model__C'])
        model = model.best_estimator_
        
        ypte = model.predict_proba(Xte)[:,1]
        ytes.extend(yte)
        yptes.extend(ypte)
        sids_tes.extend(sids_te)
        cv_ids.extend([k]*len(yte))
        
    mean_C = np.median(Cs)

    # now we do one last model
    #TODO use features selected
    
    model = Pipeline(steps=(
                ('standardizer', StandardScaler()),
                ('model', LogisticRegression(penalty='l1',
                            C=mean_C, class_weight='balanced',
                            random_state=random_state,
                            solver='saga', max_iter=10000,
                            l1_ratio=None))))
    model.fit(X, y)
    #yp_final = lasso.predict(X)
    
    return model, mean_C, yptes, ytes, sids_tes, cv_ids


if __name__=='__main__':
    random_state = 2020
    ncv = 10
    
    Xtotal = pd.read_csv('X.csv')
    ytotal = pd.read_csv('y.csv')
    info = pd.read_csv('info.csv')
    
    window_times = sorted(set(info.window_time_second))
    ynames = list(ytotal.columns)
    
    ytotal['new_spike_rate'] = ytotal['new_spike_rate']/60.
    ytotal['new_sz_burden'] = ytotal['new_sz_burden']/100.
    ytotal['new_iic_burden'] = ytotal['new_iic_burden']/100.
    
    models = {}
    ys = {}
    yps = {}
    sids = {}
    for wt in window_times:
        for yn in ynames:
            print(wt, yn)
            ids = info.window_time_second==wt
            X = Xtotal.values[ids]
            y = ytotal[yn][ids].values
            sid = info['sid'][ids].values
            
            notnanids = (~np.any(np.isnan(X), axis=1)) & (~np.isnan(y))
            X = X[notnanids]
            y = y[notnanids]
            sid = sid[notnanids]
            
            y = (y>=0.1).astype(int)
            model, mean_C, yptes, ytes, sid_tes, cv_ids = myfit(X, y, sid, ncv=ncv, random_state=random_state)
            
            models[(wt,yn)] = model
            ys[(wt,yn)] = ytes
            yps[(wt,yn)] = yptes
            sids[(wt,yn)] = sid_tes
            
    for wt in window_times:
        for yn in ynames:
            y = ys[(wt,yn)]
            yp = yps[(wt,yn)]
            print(wt, yn)
            print(roc_auc_score(y, yp))
