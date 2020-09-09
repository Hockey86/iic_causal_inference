from itertools import product
import os
import pickle
import numpy as np
from scipy.stats import binom
from scipy.special import expit as sigmoid
from scipy.special import logit
from statsmodels.tsa.arima_model import ARMA
from hashlib import md5
import pystan


def rmse(x, y):
    return np.sqrt(np.mean((x-y)**2))


class BaseSimulator(object):
    def load_model(self, save_path):
        print('loading model from %s'%save_path)
        with open(save_path, 'rb') as f:
            self.stan_model, self.fit_res, self.ma_models = pickle.load(f)
        return self

    def score(self, D, E, Ep=None, cluster=None, Ncluster=None, method='loglikelihood', TstRMSE=8):
        """
        E: [0-1], list of arrays, with different lengths. Each element has shape (T[i],)
        Ep: [0-1], list of arrays, with different lengths. Each element has shape (Nsim, T[i])
        """
        N = len(E)
        assert len(D)==N, 'len(E)!=len(D)'
        if Ep is not None:
            assert len(Ep)==N, 'len(E)!=len(E predicted)'
        available_metrics = ['loglikelihood', 'stRMSE']
        assert method in available_metrics, 'Unknown method: %s. Available: %s'%(method, str(available_metrics))

        metrics = []
        if method=='loglikelihood':
            for i in range(N):
                Ei = E[i]
                Epi = Ep[i]
                metric = []
                for j in range(len(Epi)):
                    goodids = Ei>=0
                    metric.append( np.mean(binom.logpmf(Ei[goodids], self.W, np.clip(Epi[j][goodids],1e-6,1-1e-6))) )
                metrics.append(np.array(metric))
            
        elif method=='stRMSE':
            if hasattr(self, 'T0'):
                T0 = self.T0
            else:
                T0 = 0
            for i in range(N):
                Ei = E[i]
                Di = D[i]
                if len(Ei)-TstRMSE<=T0:
                    continue
                metric = []
                for t in range(T0, len(Ei)-TstRMSE):
                    if np.any(Ei[t-T0:t+TstRMSE]<0):
                        metric.append([np.nan]*500)  #TODO assume 500
                    else:
                        Ep = self.predict([Di[t-T0:t+TstRMSE]], cluster[[i]], Ncluster=Ncluster, Pstart=Ei[t-T0:t].reshape(1,-1)/self.W)
                        metric.append([rmse(Ei[t:t+TstRMSE]/self.W, Ep[0][k,T0:]) for k in range(len(Ep[0]))])
                metrics.append(np.nanmean(metric, axis=0))
        metrics = np.array(metrics)  # (#pt, #posterior sample)

        return metrics


class BaselineSimulator(BaseSimulator):
    def __init__(self, Tinit, W, max_iter=1000, random_state=None):
        self.Tinit = Tinit
        self.W = W
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, D, E):
        return self

    def predict(self, D, E):
        if self.random_state is None:
            self.random_state = np.random.randint(10000)
        np.random.seed(self.random_state)

        Nsample = self.max_iter//2
        Eps = []
        for i in range(len(D)):
            # first decide which value to carry forward
            # it should be the (Tinit-1)-th, but it can be NaN, search backwards until non-NaN
            for tt in range(self.Tinit-1,-1,-1):
                if not np.isnan(E[i][tt]):
                    break
            tt = tt+1
            Ep = []
            for j in range(Nsample):
                Ep_ = np.random.binomial(self.W, E[i][tt-1], size=len(D[i])-tt)
                Ep_ = Ep_/self.W
                Ep_ = np.r_[E[i][:tt], Ep_]
                Ep.append(Ep_)

            Eps.append(np.array(Ep))
        return Eps


class Simulator(BaseSimulator):
    def __init__(self, stan_model_path, W, K_MA, max_iter=1000, T0=0, random_state=None):
        self.stan_model_path = stan_model_path
        self.W = W
        self.max_iter = max_iter
        self.T0 = T0
        self.random_state = random_state
        self.K_MA = K_MA

    def _get_stan_model(self, model_path):
        with open(model_path, 'r') as f:
            model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if model_path is None:
            cache_fn = 'cached-model-%s.pkl'%code_hash
        else:
            model_dir = os.path.dirname(model_path)
            model_path = os.path.basename(model_path)
            cache_fn = os.path.join(model_dir, 'cached-model-%s-%s.pkl'%(model_path, code_hash))
        try:
            stan_model = pickle.load(open(cache_fn, 'rb'))
        except:
            stan_model = pystan.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f:
                pickle.dump(stan_model, f)
        else:
            print("Using cached StanModel")
        return stan_model

    def _pad_to_same_length(self, D, E=None):
        maxT = np.max([len(x) for x in D])
        D2 = []
        if E is not None:
            P2 = []
            E2 = []
        for i in range(len(D)):
            if E is not None:
                pp = E[i].astype(float)
                pp[pp==-1] = np.nan
                P2.append( np.r_[pp/self.W, np.zeros(maxT-len(E[i]))+np.nan] )
                E2.append( np.r_[E[i], np.zeros(maxT-len(E[i]), dtype=int)-1] )
            D2.append( np.r_[D[i], np.zeros((maxT-len(D[i]), D[i].shape[1]))] )
        if E is not None:
            E2 = np.array(E2)
            P2 = np.array(P2)
        D2 = np.array(D2)

        if E is None:
            return D2
        else:
            return D2, E2, P2

    def fit(self, D_, E_, cluster,save_path=None):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T[i],ND)
        E: IIC burden, [0-1], list of arrays, with different lengths. Each element has shape (T[i],)
        cluster: Cluster assignment for the patients has shape(N,), with values [0,1,2...]
        """
        ## pad to same length
        D, E, P = self._pad_to_same_length(D_, E_)

        self.N = len(D)
        self.T = D.shape[1]
        self.ND = D.shape[-1]
        self.Ncluster = len(set(cluster.flatten()))
        Ts = np.array([np.sum(~np.isnan(x)) for x in P])

        E_flatten = E[:,self.T0:].flatten()
        not_empty_ids = np.where(E_flatten!=-1)[0]
        not_empty_num = len(not_empty_ids)
        E_flatten_nonan = E_flatten[not_empty_ids]
        #E_flatten_nonan = np.clip(E_flatten_nonan, 1, self.W-1)

        # generate sample weights that balances different lengths
        sample_weights = np.zeros_like(E[:,self.T0:]) + 1/(Ts-self.T0).reshape(-1,1)#
        sample_weights = sample_weights.flatten()[not_empty_ids]
        sample_weights = sample_weights/sample_weights.mean()

        ## load model
        self.stan_model = self._get_stan_model(self.stan_model_path)

        ## feed data

        P2 = np.clip(P, 1e-6, 1-1e-6)
        P2[np.isnan(P2)] = -1
        data_feed = {'W':self.W,
                     'N':self.N,
                     'T':self.T,
                     'T0':self.T0,
                     'ND':self.ND,
                     #'NC':C.shape[-1],
                     'not_empty_num':not_empty_num,
                     'not_empty_ids':not_empty_ids+1,  # +1 for stan
                     'sample_weights':sample_weights,
                     'Eobs_flatten_nonan':E_flatten_nonan,
                     #'Pobs':P2,
                     'D':D.transpose(1,0,2),  # because matrix[N,ND] D[T];
                     #'C':C,
                     'A_start':logit(P2[:,:self.T0]),  # assumes the first T0 steps contains no nan
                     'NClust':len(set(cluster)),
                     'cluster':cluster+1,  # +1 for stan
                     }

        ## sampling
        if self.random_state is None:
            self.random_state = np.random.randint(10000)

        self.fit_res = self.stan_model.sampling(data=data_feed,
                                       iter=self.max_iter, verbose=True,
                                       chains=1, seed=self.random_state) #print(self.fit_res.stansummary(pars=['mu_mu','sigma_mu','sigma_sigma','sigma_alpha', 'sigma_b']))
        
        # fit MA to residual
        Ep = self.predict(D_, cluster, Pstart=P2[:,:self.T0], MA=False)
        self.ma_models = []
        for i in range(len(Ep)):
            try:
                tti = Ep[i].shape[1]
                ids = ~np.isnan(P[i][:tti])
                if ids.sum()<10:
                    raise ValueError
                Ap = logit(np.clip(Ep[i].mean(axis=0), 1e-6, 1-1e-6))
                residual = logit(P2[i][:tti][ids]) - Ap[ids]
                ma_model = ARMA(residual, order=(0, self.K_MA))
                ma_model = ma_model.fit(disp=False)
                self.ma_models.append(ma_model)
            except Exception as ee:
                self.ma_models.append(None)
                
        # save
        if save_path is None:
            save_path = 'model_fit.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([self.stan_model, self.fit_res, self.ma_models], f)

        return self

    def predict(self, D, cluster, Ncluster=None, Pstart=None, MA=True):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T[i],ND)
        training:

        returns:
        P: simulated IIC burden, list of arrays, with different lengths. Each element has shape (Nsample, T[i])
        """

        # load prediction model
        self.predict_stan_model = self._get_stan_model(self.stan_model_path.replace('.stan', '_predict.stan'))

        model_type = os.path.basename(self.stan_model_path).split('_')[-1].replace('.stan','')
        N = len(D)
        ND = D[0].shape[-1]
        
        # set model-specific parameters as input data
        if model_type in ['AR1', 'PAR1']:
            pars = ['a0','a1','b']
            pars_shape = [(N,), (N,), (N,ND)]
            pars_shape2 = [1,1,ND]
        #elif model_type in ['NBAR1']:
        elif model_type in ['AR2', 'PAR2']:
            pars = ['a0','a1','a2','b']
            pars_shape = [(N,), (N,), (N,), (N,ND)]
            pars_shape2 = [1,1,1,ND]
        #elif model_type in ['NBAR2']:
        elif model_type=='lognormal':
            pars = ['mu', 'alpha', 'sigma', 'b']#'t0',
            pars_shape = [(N,), (N,), (N,), (N,ND)]
            pars_shape2 = [1,1,1,ND]
        elif model_type=='lognormalAR1':
            pars = ['a0', 'a1', 'mu', 'sigma', 'b']#'t0',
            pars_shape = [(N,), (N,), (N,), (N,), (N,ND)]
            pars_shape2 = [1,1,1,1,ND]
        elif model_type=='lognormalAR2':
            pars = ['a0', 'a1', 'a2', 'mu', 'sigma', 'b']#'t0',
            pars_shape = [(N,), (N,), (N,), (N,), (N,), (N,ND)]
            pars_shape2 = [1,1,1,1,1,ND]
        else:
            raise NotImplementedError(self.stan_model_path)

        df = self.fit_res.to_dataframe(pars=pars)
        Nsample = len(df)
        data_feed2 = {'N_sample':Nsample}  # data_feed2 is model-specific

        # the following is a general code to convert['par[?,?]'] into array of shape (?,?),
        # and then assign to data_feed['par']
        # since it is very general, so it is not easy to read
        for pi, par in enumerate(pars):
            shape = pars_shape[pi]
            var = np.zeros((Nsample,)+shape)
            for ind in product(*[range(1,x+1) for x in shape]):
                key = '%s[' + ','.join(['%d']*len(ind)) + ']'
                val = (par,) + ind
                inds = [slice(None)]*var.ndim
                for ii, jj in enumerate(ind):
                    inds[ii+1] = jj-1
                var[tuple(inds)] = df[key%val].values
            data_feed2[par] = var

        # also add AR-specific initial values
        if Pstart is not None and 'AR' in model_type:
            if 'PAR' in model_type or 'NBAR' in model_type:
                func = np.log
            else:
                func = logit
            A_start = func(np.clip(Pstart, 1e-6, 1-1e-6))
            data_feed2['A_start'] = A_start
            data_feed2['T0'] = Pstart.shape[-1]

        # set model-unspecific input data
        Ts = [len(x) for x in D]
        D = self._pad_to_same_length(D)
        N, T, ND = D.shape
        #assert self.ND==ND, 'ND is not the same as in fit'
        #assert T>self.T0, 'T<=T0'

        if hasattr(self, 'Ncluster'):
            Ncluster = self.Ncluster
        data_feed = {'W':self.W,
                     'N':N,
                     'T':T,
                     'ND':ND,
                     'D':D.transpose(1,0,2),  # because matrix[N,ND] D[T];
                     'NClust':Ncluster,
                     'cluster':cluster+1,  # +1 for stan
                     }

        # combine model-specific and model-unspecific input data
        data_feed.update(data_feed2)

        # sample without inferring parameters
        self.predict_res = self.predict_stan_model.sampling(data=data_feed,
                                       iter=1, verbose=False,
                                       chains=1, seed=self.random_state,
                                       algorithm = "Fixed_param")

        df_res = self.predict_res.to_dataframe(pars=['P_output'])

        cols = []
        for i in range(Nsample):
            for j in range(N):
                for k in range(T):
                    cols.append('P_output[%d,%d,%d]'%(i+1,j+1,k+1))
        P = df_res[cols].values[0]
        P = P.reshape(Nsample, N, T)

        P = [P[:,i,:Ts[i]] for i in range(N)]

        if MA:
            for i in range(len(P)):
                if self.ma_models[i] is None:
                    continue
                tti = P[i].shape[1]
                Ap = logit(np.clip(P[i], 1e-6, 1-1e-6))
                for j in range(len(P[i])):
                    ww = np.random.randn(tti)*np.sqrt(self.ma_models[i].sigma2)
                    ww2 = np.convolve(ww, self.ma_models[i].params[1:], mode='same')+self.ma_models[i].params[0]
                    P[i][j] = sigmoid(Ap[j] + ww2)
        return P
        
