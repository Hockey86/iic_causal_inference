from itertools import product
import os
import pickle
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import logit
from hashlib import md5
import pystan


class Simulator(object):
    def __init__(self, stan_model_path, W, max_iter=1000, T0=0, random_state=None):
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
        
    def fit(self, D, E, save_path=None):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T,ND)
        E: IIC burden, [0-1], list of arrays, with different lengths. Each element has shape (T,)
        """
        ## pad to same length
        D, E, P = self._pad_to_same_length(D,E)
        
        self.N = len(D)
        self.T = D.shape[1]
        self.ND = D.shape[-1]
        Ts = np.array([np.sum(~np.isnan(x)) for x in P])
        
        E_flatten = E[:,self.T0:].flatten()
        not_empty_ids = np.where(E_flatten!=-1)[0]
        not_empty_num = len(not_empty_ids)
        E_flatten_nonan = E_flatten[not_empty_ids]

        # generate sample weights that balances different lengths
        sample_weights = np.zeros_like(E[:,self.T0:]) + 1/(Ts-self.T0).reshape(-1,1)#
        sample_weights = sample_weights.flatten()[not_empty_ids]
        sample_weights = sample_weights/sample_weights.mean()
        
        ## load model
        self.stan_model = self._get_stan_model(self.stan_model_path)

        ## feed data
        
        Pstart = np.clip(P[:,:self.T0], 1e-6, 1-1e-6)
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
                     'D':D.transpose(1,0,2),  # because matrix[N,ND] D[T];
                     #'C':C,
                     'A_start':logit(Pstart),
                     }

        ## sampling
        if self.random_state is None:
            self.random_state = np.random.randint(10000)
            
        self.fit_res = self.stan_model.sampling(data=data_feed,
                                       iter=self.max_iter, verbose=True,
                                       chains=1, seed=self.random_state)
        #print(self.fit_res.stansummary(pars=['mu_mu','sigma_mu','sigma_sigma','sigma_alpha', 'sigma_b']))

        # save
        if save_path is None:
            save_path = 'model_fit.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([self.stan_model, self.fit_res], f)
        """
        with open(save_path, 'rb') as f:
            self.stan_model, self.fit_res = pickle.load(f)
        """
            
        return self
        
    def predict(self, D, training=False, Pstart=None):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T,ND)
        training:
        
        returns:
        P: simulated IIC burden, list of arrays, with different lengths. Each element has shape (Nsample, T)
        """
        
        # load prediction model
        self.predict_stan_model = self._get_stan_model(self.stan_model_path.replace('.stan', '_predict.stan'))
        
        if training:
            # set model-specific parameters as input data
            if self.stan_model_path.endswith('_AR1.stan'):
                pars = ['a0','a1','b']
                pars_shape = [(self.N,), (self.N,), (self.N, self.ND)]
            elif self.stan_model_path.endswith('_AR2.stan'):
                pars = ['a0','a1','a2','b']
                pars_shape = [(self.N,), (self.N,), (self.N,), (self.N, self.ND)]
            elif self.stan_model_path.endswith('_lognormal.stan'):
                pars = ['t0', 'mu', 'alpha', 'sigma', 'b']
                pars_shape = [(self.N,), (self.N,), (self.N,), (self.N,), (self.N, self.ND)]
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
        
        else:
            #if self.stan_model_path.endswith('_AR1.stan'):
            #    pars = ['mu_a0','mu_a1','mu_b']
            raise NotImplementedError('training == False')
            
        
        # also add AR-specific initial values
        if self.stan_model_path.endswith('_AR1.stan') or \
           self.stan_model_path.endswith('_AR2.stan'):
            A_start = logit(np.clip(Pstart, 1e-6, 1-1e-6))
            data_feed2['A_start'] = A_start
            data_feed2['T0'] = Pstart.shape[-1]
            
        # set model-unspecific input data
        Ts = [len(x) for x in D]
        D = self._pad_to_same_length(D)
        N, T, ND = D.shape
        assert self.ND==ND, 'ND is not the same as in fit'
        assert T>self.T0, 'T<=T0'
        
        data_feed = {'W':self.W,
                     'N':N,
                     'T':T,
                     'ND':self.ND,
                     'D':D.transpose(1,0,2),  # because matrix[N,ND] D[T];
                     }
                     
        # combine model-specific and model-unspecific input data
        data_feed.update(data_feed2)
                     
        # sample without inferring parameters
        self.predict_res = self.predict_stan_model.sampling(data=data_feed,
                                       iter=1, verbose=False,
                                       chains=1, seed=self.random_state,
                                       algorithm = "Fixed_param")
        
        df_res = self.predict_res.to_dataframe(pars=['P_output'])
        
        P = np.zeros((Nsample, N, T))
        for i in range(Nsample):
            for j in range(N):
                for k in range(T):
                    P[i,j,k] = df_res['P_output[%d,%d,%d]'%(i+1,j+1,k+1)].values[0]
        
        P = [P[:,i,:Ts[i]] for i in range(len(P))]
            
        return P

