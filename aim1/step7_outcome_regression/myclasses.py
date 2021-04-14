#!/usr/bin/env python
# coding: utf-8
from itertools import combinations
from collections import Counter
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.ensemble import RandomForestClassifier
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


def get_sample_weights(y, class_weight='balanced', prior_count=0):
    assert y.min()==0 ## assume y=[0,1,...]
    K = y.max()+1
    class_weights = {k:1./(np.sum(y==k)+prior_count) for k in range(K)}
    sw = np.array([class_weights[yy] for yy in y])
    sw = sw/np.mean(sw)
    return sw
    
    
class MyLogisticRegression(LogisticRegression):
    """
    Allows bounds
    No intercept
    """
    def __init__(self, class_weight=None, tol=1e-6, C=1.0, l1_ratio=0., random_state=None, max_iter=1000, bounds=None):
        super().__init__(penalty='elasticnet', dual=False, tol=tol, C=C,
                 fit_intercept=False, intercept_scaling=1, class_weight=class_weight,
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
            out_penalty = 0.5*alpha*(1 - l1_ratio)*np.sum(w**2) + alpha*l1_ratio*np.sum(np.abs(w))
            grad_penalty = alpha*(1-l1_ratio)*w+alpha*l1_ratio*np.sign(w)# ,0]
            return out+out_penalty, grad+grad_penalty
        
        y2 = np.array(y)
        y2[y2==0] = -1
        w0 = np.random.randn(X.shape[1])/10#, 0.]
        #if self.bounds is None:
        #    method = 'BFGS'
        #else:
        #    method = 'L-BFGS-B'
        method = None
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight)
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))
        self.opt_res = minimize(
            func, w0, method=method, jac=True,
            args=(X, y2, 1./self.C, self.l1_ratio, sample_weight),
            bounds=self.bounds,
            options={"gtol": self.tol, "maxiter": self.max_iter}
        )
        coef_ = self.opt_res.x#[:-1]
        #intercept_ = self.opt_res.x[-1]
        
        self.coef_ = coef_.reshape(1,-1)
        self.intercept_ = np.zeros(1)#intercept_.reshape(1,)
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


class MyRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, impute_KNN_K=10, binary_indicator=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        self.impute_KNN_K = impute_KNN_K
        self.binary_indicator = binary_indicator

    def fit(self, X, y, sample_weight=None):
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
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = np.array(X)
        # standardize
        X[:,~self.binary_indicator] = (X[:,~self.binary_indicator]-self.Xmean)/self.Xstd
        # impute missing value
        if np.any(np.isnan(X)):
            X = self.imputer.transform(X)
            X[:,self.binary_indicator] = (X[:,self.binary_indicator]>0.5).astype(float)
        return super().predict(X)

    def predict_proba(self, X):
        X = np.array(X)
        # standardize
        X[:,~self.binary_indicator] = (X[:,~self.binary_indicator]-self.Xmean)/self.Xstd
        # impute missing value
        if np.any(np.isnan(X)):
            X = self.imputer.transform(X)
            X[:,self.binary_indicator] = (X[:,self.binary_indicator]>0.5).astype(float)
        return super().predict_proba(X)


def tricube(x):
    return (1-np.abs(x)**3)**3
    
    
def decide_neighbor(X, Xref, n_neighbors, sigma=None, relative=True, weight=False):
    dists = np.sqrt(cdist(X, Xref, metric='sqeuclidean')/X.shape[1])
    max_dist = dists.max()
    if sigma is None:
        sigma_ = np.inf
    else:
        if relative:
            q1, q3 = np.percentile(dists.flatten(), (2.5, 97.5))
            sigma_ = q1 + (q3-q1)*sigma
        else:
            sigma_ = sigma
    # if weight, make the criteria more relaxed
    if weight:
        sigma_ *= 2
        n_neighbors *= 2
    inside_sigma = dists<=sigma_
    
    neighbors = []
    weights = []
    for i in range(len(dists)):
        if inside_sigma[i].sum()<=n_neighbors:
            neighbors.append(np.where(inside_sigma[i])[0])
        else:
            neighbors.append(np.argsort(dists[i])[:n_neighbors])
        if weight:
            weights.append(tricube(dists[i][neighbors[-1]]/max_dist))
        
    if weight:
        return neighbors, weights
    else:
        return neighbors
    
    
class LocalModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        #TODO determine good features
        self.good_ids = [
            0,1,2,3,4,5,6,7,8,9,
            10,11,12,17,18,19,20,24,
            25,27,28,30,32,33,34,36,37]
        
        self.X = np.array(X[:,self.good_ids])
        self.binary_idx = np.array([set(self.X[:,i][~np.isnan(self.X[:,i])])==set([0,1]) for i in range(self.X.shape[1])])
        self.cont_idx = np.where(~self.binary_idx)[0]
        self.binary_idx = np.where(self.binary_idx)[0]

        # standardize
        self.Xmean = np.nanmean(self.X[:,self.cont_idx], axis=0)
        self.Xstd = np.nanstd(self.X[:,self.cont_idx], axis=0)
        self.X[:,self.cont_idx] = (self.X[:,self.cont_idx]-self.Xmean) / self.Xstd
        
        # impute
        self.imputer = KNNImputer(n_neighbors=self.impute_KNN_K).fit(self.X)
        Xhasmissing = np.any(np.isnan(self.X))
        if Xhasmissing:
            self.X = self.imputer.transform(self.X)
            self.X[:,self.binary_idx] = (self.X[:,self.binary_idx]>0.5).astype(float)
            
        self.le = LabelEncoder()
        self.le.fit(y)
        self.y = self.le.transform(y)
        self.classes_ = self.le.classes_
    
        return self
    
    def predict(self, X):
        yp = self.predict_proba(X)
        yp = self.le.inverse_transform(np.argmax(yp, axis=1))
        return yp
        
        
class KNN(LocalModel):
    def __init__(self, n_neighbors=10, sigma=None, relative=True, impute_KNN_K=10):
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.relative = relative
        self.impute_KNN_K = impute_KNN_K
    
    def predict_proba(self, X):
        X = np.array(X[:,self.good_ids])

        # standardize
        X[:,self.cont_idx] = (X[:,self.cont_idx]-self.Xmean)/self.Xstd
        # impute missing value
        if np.any(np.isnan(X)):
            X = self.imputer.transform(X)
            X[:,self.binary_idx] = (X[:,self.binary_idx]>0.5).astype(float)
        
        neighbors = decide_neighbor(X, self.X, self.n_neighbors, sigma= self.sigma, relative= self.relative)
        yp = []
        Nclass = len(self.classes_)
        for i in range(len(X)):
            if len(neighbors[i])<=3:
                yp_ = [0]*Nclass
                idx = np.random.choice(
                        np.arange(Nclass), replace=False,
                        p=[np.mean(self.y==l) for l in np.arange(Nclass)])
                yp_[idx] = 1
                yp.append(yp_)
            else:
                counter = Counter(self.y[neighbors[i]])
                yp.append( [counter[k] for k in range(len(self.classes_))] )
        yp = np.array(yp)
        yp = yp*1./yp.sum(axis=1, keepdims=True)
        return yp


class LOWESS(LocalModel):
    def __init__(self, model, n_neighbors=10, sigma=None, relative=True, impute_KNN_K=10):
        self.model = model
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.relative = relative
        self.impute_KNN_K = impute_KNN_K
    
    def predict_proba(self, X):
        X = np.array(X[:,self.good_ids])
        # standardize
        X[:,self.cont_idx] = (X[:,self.cont_idx]-self.Xmean)/self.Xstd
        # impute missing value
        if np.any(np.isnan(X)):
            X = self.imputer.transform(X)
            X[:,self.binary_idx] = (X[:,self.binary_idx]>0.5).astype(float)
        
        neighbors, weights = decide_neighbor(X, self.X, self.n_neighbors, sigma= self.sigma, relative= self.relative, weight=True)
        yp = []
        Nclass = len(self.classes_)
        for i in range(len(X)):
            if len(neighbors[i])>=10 and len(set(self.y[neighbors[i]]))==1:
                yp_ = [0]*Nclass
                yp_[self.y[neighbors[i][0]]] = 1
                yp.append(yp_)
            elif len(neighbors[i])<=10:
                yp_ = [0]*Nclass
                idx = np.random.choice(
                        np.arange(Nclass), replace=False,
                        p=[np.mean(self.y==l) for l in np.arange(Nclass)])
                yp_[idx] = 1
                yp.append(yp_)
            else:
                local_model = self.model.fit(self.X[neighbors[i]], self.y[neighbors[i]], sample_weight=weights[i])
                yp.append( local_model.predict_proba(X[i].reshape(1,-1))[0] )
        yp = np.array(yp)
        return yp
        
