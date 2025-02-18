from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binom, spearmanr
from scipy.special import expit as sigmoid
from scipy.special import logit
#from statsmodels.tsa.arima_model import ARMA
from tqdm import tqdm
from hashlib import md5
import pystan


def rmse(x, y):
    return np.sqrt(np.mean((x-y)**2))


class BaseSimulator(object):
    def load_model(self, path):
        print('loading model from %s'%path)
        with open(path, 'rb') as f:
            self.stan_model, self.fit_res_df, self.Ncluster = pickle.load(f)#, self.ma_models
        return self

    def save_model(self, path):
        # save dataframe to avoid big file size
        with open(path, 'wb') as f:
            pickle.dump([self.stan_model, self.fit_res_df, self.Ncluster], f)
    
    @property
    def waic(self):
        cols = [x for x in self.fit_res_df.columns if 'log_lik' in x]
        log_lik_df = self.fit_res_df[cols]
        parameters_WAIC = log_lik_df.apply(np.var,1)
        parameters_WAIC = np.sum(parameters_WAIC)
        lppd = np.sum( log_lik_df.apply(np.mean,0) )
        waic = -2 *lppd +2* parameters_WAIC
        return waic
        
    def score(self, P, Psim, method, D=None, cluster=None, Ncluster=None, Ts_stRMSE=None, Tinterval=1):
        """
        P: [0-1], list of arrays, with different lengths. Each element has shape (T[i],)
        Psim: [0-1], list of arrays, with different lengths. Each element has shape (Nsim, T[i])
        """
        N = len(P)
        assert len(Psim)==N, 'len(P)!=len(Psim)'
        #available_metrics = ['loglikelihood', 'stRMSE']
        #assert method in available_metrics, 'Unknown method: %s. Available: %s'%(method, str(available_metrics))

        metrics = []
        if method=='CI95 Coverage':
            for i in range(N):
                Pi = P[i]
                Psim_i = Psim[i]
                goodids = ~np.isnan(Pi)
                Pi = Pi[goodids]
                Psim_i = Psim_i[:,goodids]
                lb, ub = np.percentile(Psim_i, (2.5, 97.5), axis=0)
                # if in the middle, check if between the bounds
                criterion1 = (Pi>0.05) & (Pi<0.95) & (Pi<=ub) & (Pi>=lb)
                # if close to boundary, check if bounds are also close to boundary
                criterion2 = ((Pi<=0.05) & (lb<=0.05)) | ((Pi>=0.95) & (ub>=0.95))
                # either of above
                metric = np.mean( criterion1 | criterion2 )
                metrics.append(metric)

        elif method=='loglikelihood':
            for i in range(N):
                Pi = P[i]
                Psim_i = Psim[i]
                goodids = ~np.isnan(Pi)
                metric = []
                for j in range(len(Psim_i)):
                    metric.append( np.mean(binom.logpmf(np.round(Pi[goodids]*self.W).astype(int), self.W, np.clip(Psim_i[j][goodids],1e-6,1-1e-6))) )
                metrics.append(np.array(metric))
        
        elif method=='KL-divergence':
            for i in range(N):
                Pi = P[i]
                Psim_i = Psim[i]
                goodids = ~np.isnan(Pi)
                Pi = Pi[goodids]
                Psim_i = Psim_i[:,goodids]
                T = len(Pi)
                ps = np.array([np.mean((Pi>=lb)&(Pi<lb+0.1)) for lb in np.arange(0,1,0.1)])
                qs = np.array([(np.sum((Psim_i>=lb)&(Psim_i<lb+0.1), axis=1)+0.1)/(T+1) for lb in np.arange(0,1,0.1)]).T
                kl = ps*np.log(ps/qs)
                kl[:,ps==0] = np.nan
                kl[qs==0] = np.nan
                kl[np.isinf(kl)] = np.nan
                kl = np.nansum(kl, axis=1)
                metrics.append(kl)
        
        elif method=='spearmanr':
            for i in range(N):
                Pi = P[i]
                Psim_i = Psim[i]
                goodids = ~np.isnan(Pi)
                Pi = Pi[goodids]
                Psim_i = Psim_i[:,goodids]
                r = spearmanr(Pi.reshape(1,-1), Psim_i, axis=1).correlation
                r = r[0,1:]
                metrics.append(r)

        elif method=='WAIC':
            metrics = self.waic
            
        elif method=='stRMSE':
            if hasattr(self, 'AR_T0'):
                T0 = self.AR_T0
            elif hasattr(self, 'T0'):
                T0 = self.T0[0]
            if type(self)==BaselineSimulator:
                T0 = 2
            max_horizon = np.max(Ts_stRMSE)
            for i in range(N):
                Pi = P[i]
                Di = D[i]
                
                # decide initialization starts,
                # evenly choose `start_num` start locations with interval=`Tinterval`
                # from %missing<10% start locations
                start_locs = np.array([t0 for t0 in range(0, len(Pi)-T0-max_horizon) if np.mean(np.isnan(Pi[t0:t0+T0+max_horizon]))<=0.5 and np.all(~np.isnan(Pi[t0:t0+T0]))])
                start_num = len(range(0, len(Pi)-T0-max_horizon+1, Tinterval))
                start_locs = start_locs[np.arange(0, len(start_locs), len(start_locs)//start_num)]
                
                metric = {horizon:[] for horizon in Ts_stRMSE}
                for t0 in start_locs:
                    if t0==0:
                        Psim_i = Psim[i]
                    else:
                        Psim_i = self.predict([Di[t0:t0+T0+max_horizon]], cluster[[i]], Ncluster=Ncluster, Pstart=Pi[t0:t0+T0].reshape(1,-1), sid_index=[i])[0]
                    
                    for horizon in Ts_stRMSE:
                        rmse = np.sqrt(np.nanmean((Pi[t0+T0:t0+T0+horizon] - Psim_i[:,T0:T0+horizon])**2, axis=1))
                        metric[horizon].append(rmse)
                        
                metrics.append( [np.nanmean(metric[horizon], axis=0) for horizon in Ts_stRMSE] )

        metrics = np.array(metrics)  # (#pt, #posterior sample)                
        return metrics


class BaselineSimulator(BaseSimulator):
    def __init__(self, Tinit, W, max_iter=1000, random_state=None):
        self.Tinit = Tinit
        self.W = W
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, D, P, cluster):
        N = len(P)
        self.stan_model = None
        
        # generate sample weights that balances different lengths
        sample_weights = []
        for i in range(N):
            sample_weights.append([1./np.sum(~np.isnan(P[i]))]*len(P[i]))
        sw_mean = np.concatenate(sample_weights).mean()
        sample_weights = [x[0]/sw_mean for x in sample_weights]
        
        # get self.fit_res_df
        Psim = self.predict(D, cluster, Pstart=P)
        log_lik = []
        for i in range(N):
            ids = ~np.isnan(P[i])
            k = np.round(P[i][ids]*self.W).astype(int)
            p = Psim[i][:,ids]
            log_lik.append( binom.logpmf(k, self.W, np.clip(p,1e-6,1-1e-6) ) * sample_weights[i] )
        
        log_lik = np.concatenate(log_lik, axis=1)
        self.fit_res_df = pd.DataFrame(
                            data=log_lik,
                            columns=['log_lik[%d]'%(i+1,) for i in range(log_lik.shape[1])])
        return self

    def predict(self, D, cluster, Ncluster=None, sid_index=None, Pstart=None, posterior_mean=False):#, MA=True):
        if self.random_state is None:
            self.random_state = np.random.randint(10000)
        np.random.seed(self.random_state)

        Nsample = self.max_iter//2
        P_pred = []
        for i in range(len(D)):
            # first decide which value to carry forward
            # it should be the (Tinit-1)-th, but it can be NaN, search backwards until non-NaN
            for tt in range(self.Tinit-1,-1,-1):
                if not np.isnan(Pstart[i][tt]):
                    break
            tt = tt+1
            this_P_pred = []
            for j in range(Nsample):
                p_ = np.random.binomial(self.W, Pstart[i][tt-1], size=len(D[i])-tt)/self.W
                p_ = np.r_[Pstart[i][:tt], p_]
                this_P_pred.append(p_)

            P_pred.append(np.array(this_P_pred))
        return P_pred


class Simulator(BaseSimulator):
    def __init__(self, stan_model_path, W, max_iter=1000, T0=[1,1], random_state=None):
        self.stan_model_path = stan_model_path
        self.W = W
        self.max_iter = max_iter
        self.T0 = T0
        self.random_state = random_state

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

    def _pad_to_same_length(self, D, P=None):
        maxT = np.max([len(x) for x in D])
        D2 = []
        if P is not None:
            P2 = []
            E2 = []
        for i in range(len(D)):
            if P is not None:
                P2.append( np.r_[P[i], np.zeros(maxT-len(P[i]))+np.nan] )
                E2.append( np.r_[np.round(P[i]*self.W).astype(int), np.zeros(maxT-len(E[i]), dtype=int)-1] )
            D2.append( np.r_[D[i], np.zeros((maxT-len(D[i]), D[i].shape[1]))] )
        if P is not None:
            E2 = np.array(E2)
            P2 = np.array(P2)
        D2 = np.array(D2)

        if P is None:
            return D2
        else:
            return D2, E2, P2

    def convert_to_ragged_structure(self, D, P, W, cluster):
        Ts = np.array([len(x) for x in P])
        Ts_nonan = np.array([np.sum(~np.isnan(x)) for x in P])
        D2 = np.concatenate(D, axis=0)
        P2 = np.concatenate(P, axis=0)
        E2 = P2*W
        E2[np.isnan(P2)] = -1
        E2 = np.round(E2).astype(int)
        cluster2 = np.concatenate([[cluster[i]]*Ts[i] for i in range(len(Ts))])
        return D2, E2, P2, cluster2, Ts, Ts_nonan

    #TODO def fit(self, D_, P_, cluster):

    def fit_parallel(self, Ddose_, Dhalflife, P_, cluster, n_jobs=1):
        """
        Ddose_: drug dose, list of arrays, with different lengths. Each element has shape (T[i],ND)
        P_: response, [0-1], list of arrays, with different lengths. Each element has shape (T[i],)
        cluster: Cluster assignment for the patients has shape(N,), with values [0,1,2...]
        """
        ## pad to same length
        Ddose, E, P, cluster2, Ts, Ts_nonan = self.convert_to_ragged_structure(Ddose_, P_, self.W, cluster)

        self.N = len(Ddose_)
        self.ND = Ddose_[0].shape[-1]
        self.Ncluster = len(set(cluster.flatten()))

        # generate sample weights that balances different lengths
        sample_weights = np.zeros_like(P)
        cc = 0
        for i in range(self.N):
            sample_weights[cc:cc+Ts[i]] = 1/Ts_nonan[i]
            cc += Ts[i]
        sample_weights = sample_weights/sample_weights.mean()

        ## load model
        self.stan_model = self._get_stan_model(self.stan_model_path)

        # data feed
        if self.random_state is None:
            self.random_state = np.random.randint(10000)
        
        def _inner_fit(data_feed, stan_model, max_iter, random_state):
            fit_res = stan_model.sampling(data=data_feed,
                                   iter=max_iter, verbose=True,
                                   chains=1, seed=random_state)
            pars = fit_res.model_pars
            pars = [x for x in pars if x not in ['A', 'D', 'err', 'ones_b', 'pos']]
            pars = [x for x in pars if 'transform' not in x.lower() and not x.lower().startswith('tmp')]
            fit_res_df = fit_res.to_dataframe(pars=pars)
            return fit_res, fit_res_df
        
        data_feeds = []
        for j in range(self.Ncluster):
            ids = cluster==j
            ids2 = cluster2==j
            data_feeds.append( {
                'W':self.W, 'N':np.sum(ids), 'ND':self.ND,
                'AR_p':self.T0[0],# 'MA_q':self.T0[1],

                'total_len':len(E[ids2]),
                'patient_lens':Ts[ids],
                'Eobs':E[ids2],
                'sample_weights':sample_weights[ids2],
                'Ddose': Ddose[ids2],  # because matrix[N,ND] Ddose[T];
                'Dhalflife_mean':Dhalflife[:,0],
                'Dhalflife_lb':Dhalflife[:,1],
                'Dhalflife_ub':Dhalflife[:,2],

                'NClust':1,
                'cluster':np.zeros_like(cluster[ids]) +1,  # +1 because of Stan
            } )
            
        ## sampling
        fit_res = Parallel(n_jobs=n_jobs)(delayed(_inner_fit)(
                    data_feeds[j],
                    self.stan_model,
                    self.max_iter,
                    self.random_state) for j in range(self.Ncluster))

        for j in range(self.Ncluster):
            print('\nCluster', j)
            if self.N>100:
                print(fit_res[j][0].stansummary(pars=['sigma_alpha']))
            else:
                print(fit_res[j][0].stansummary(pars=['alpha0']))

        # combine
        self.fit_res_df = []
        for j in tqdm(range(self.Ncluster)):
            fit_res_df = fit_res[j][1]
            col_maps = {}
            for k in ['sigma_alpha', 'sigma_lambda0', 'sigma_t0', 'sigma_sigma0','sigma_err']:
                col_maps[f'{k}[1]'] = f'{k}[{j+1}]'
            for k in range(self.ND):
                col_maps[f'sigma_beta[1,{k+1}]'] = f'sigma_beta[{j+1},{k+1}]'
                col_maps[f'sigma_halflife[1,{k+1}]'] = f'sigma_halflife[{j+1},{k+1}]'
            ids = np.where(cluster==j)[0]
            ids2 = np.where(cluster2==j)[0]
            for k in ['t0', 'sigma0', 'lambda0', 'alpha0', 'alpha1', 'alpha2']:
                for n in range(len(ids)):
                    col_maps[f'{k}[{n+1}]'] = f'{k}[{ids[n]+1}]'
            #for n in range(len(ids2)):
            #    col_maps[f'log_lik[{n+1}]'] = f'log_lik[{ids2[n]+1}]'
            for k in range(self.ND):
                for n in range(len(ids)):
                    col_maps[f'beta[{n+1},{k+1}]'] = f'beta[{ids[n]+1},{k+1}]'
                    col_maps[f'halflife[{n+1},{k+1}]'] = f'halflife[{ids[n]+1},{k+1}]'
            fit_res_df = fit_res_df.rename(columns=col_maps)
            self.fit_res_df.append(fit_res_df)
        self.fit_res_df = pd.concat(self.fit_res_df, axis=1)

        for col in self.fit_res_df:
            if col.startswith('halflife['):
                Did = int(col.split(',')[-1][:-1])-1
                self.fit_res_df[col] = self.fit_res_df[col]*(Dhalflife[Did,2]-Dhalflife[Did,1])+Dhalflife[Did,1]
        
        return self

    #TODO def predict_incr
    
    def predict(self, Ddose, cluster, Ncluster=None, sid_index=None, Astart=None, Dstart=None, posterior_mean=False, verbose=True, return_AD=False):#, MA=True):
        """
        Ddose: drug dose, list of arrays, with different lengths. Each element has shape (T[i],ND)
        cluster:

        returns:
        P: simulated IIC burden, list of arrays, with different lengths. Each element has shape (Nsample, T[i])
        """

        model_type = os.path.basename(self.stan_model_path).replace('model_','').replace('.stan','')
        N = len(Ddose)
        ND = Ddose[0].shape[-1]
        if Ncluster is None:
            if hasattr(self, 'Ncluster'):
                Ncluster = self.Ncluster
            else:
                Ncluster = len(set(cluster))

        # set model-specific parameters as input data
        if 'ARMA' in model_type:
            pars = ['alpha0','alpha1', 'alpha2','beta', 'halflife', 'sigma_err']
            pars_shape = [(N,), (N,), (N,), (N,ND),(N,ND), (Ncluster,)]
            pars_shape0_is_N = [True, True, True, True, True, False]
            #if 'lognormal' in model_type:
            pars.extend(['t0','sigma0','lambda0'])
            pars_shape.extend([(N,), (N,),(N,)])
            pars_shape0_is_N.extend([True, True, True])
        else:
            raise NotImplementedError(self.stan_model_path)

        #df = self.fit_res.to_dataframe(pars=pars)
        df = self.fit_res_df
        if posterior_mean:
            values = df.values
            values[values>1e6] = np.nan
            values = np.nanmean(values, axis=0)
            values[np.isnan(values)] = 0
            df.loc[0] = values
            df = df[:1]
        Nsample = len(df)
        data_feed2 = {'N_sample':Nsample}  # data_feed2 is model-specific

        # the following is a general code to convert['par[?,?]'] into array of shape (?,?),
        # and then assign to data_feed['par']
        # since it is very general, so it is not easy to read
        for pi, par in enumerate(pars):
            shape = pars_shape[pi]
            if len(shape)>0:
                var = np.zeros((Nsample,)+shape)
                for ind in product(*[range(1,x+1) for x in shape]):
                    key = '%s[' + ','.join(['%d']*len(ind)) + ']'
                    if pars_shape0_is_N[pi] and sid_index is not None:
                        val = (par,) + (sid_index[ind[0]-1]+1,)+ind[1:]
                    else:
                        val = (par,) + ind
                    inds = [slice(None)]*var.ndim
                    for ii, jj in enumerate(ind):
                        inds[ii+1] = jj-1
                    var[tuple(inds)] = df[key%val].values
            else:
                var = df[par].values
            data_feed2[par] = var

        # also add AR-specific initial values
        if 'AR' in model_type:
            if Astart is not None:
                #func = logit
                #data_feed2['Astart'] = func(np.clip(Pstart, 1e-6, 1-1e-6))
                data_feed2['Astart'] = Astart
            if Dstart is not None:
                data_feed2['Dstart'] = Dstart

        # set model-unspecific input data
        Ts = [len(x) for x in Ddose]
        Ddose = self._pad_to_same_length(Ddose)
        N, T, ND = Ddose.shape
        #assert self.ND==ND, 'ND is not the same as in fit'
        #assert T>self.T0, 'T<=T0'

        data_feed = {'W':self.W,
                     'N':N,
                     'T':T,
                     'AR_p':self.T0[0],
                     'MA_q':self.T0[1],
                     'ND':ND,
                     'Ddose':Ddose.transpose(1,0,2),  # because matrix[N,ND] Ddose[T];
                     'NClust':Ncluster,
                     'cluster':np.array(cluster),
                     }

        # combine model-specific and model-unspecific input data
        data_feed.update(data_feed2)

        exec(f'from stan_models.model_{model_type}_predict import predict', globals())
        P, latent = predict(data_feed, random_state=self.random_state, verbose=verbose)
        A = latent['A']
        D = latent['D']

        """
        # sample without inferring parameters
        self.predict_stan_model = self._get_stan_model(self.stan_model_path.replace('.stan', '_predict.stan'))
        data_feed['cluster'] += 1  # +1 for stan
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
        """

        P = [P[:,i,:Ts[i]] for i in range(N)]
        A = [A[:,i,:Ts[i]] for i in range(N)]
        D = [D[:,i,:Ts[i]] for i in range(N)]

        """
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
        """
        if return_AD:
            return P, A, D
        else:
            return P
