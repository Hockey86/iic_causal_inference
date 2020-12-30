#!/usr/bin/env python
# coding: utf-8
from itertools import combinations
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
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
