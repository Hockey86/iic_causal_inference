import pickle
import numpy as np
from scipy.special import expit as sigmoid
from hashlib import md5


class Simulator(object):
    def __init__(self, stan_model_path, W, random_state=None):
        self.stan_model_path = stan_model_path
        self.W = W
        self.random_state = random_state
        
    def fit(self, D, E, save_path=None, stan_model_path=None):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T,ND)
        E: IIC burden, [0-1], list of arrays, with different lengths. Each element has shape (T,)
        """
        ## pad to same length
        maxT = np.max([len(x) for x in D])
        D2 = []
        P2 = []
        E2 = []
        for i in range(len(D)):
            pp = E[i].astype(float)
            pp[pp==-1] = np.nan
            P2.append( np.r_[pp/self.W, np.zeros(maxT-len(E[i]))+np.nan] )
            E2.append( np.r_[E[i], np.zeros(maxT-len(E[i]), dtype=int)-1] )
            D2.append( np.r_[D[i], np.zeros((maxT-len(D[i]), D[i].shape[1]))] )
        E = np.array(E2)
        P = np.array(P2)
        D = np.array(D2)
        
        N = len(D)
        T = D.shape[1]
        self.ND = D.shape[-1]
        Ts = np.array([np.sum(~np.isnan(x)) for x in P])
        
        E_flatten = E.flatten()
        not_empty_ids = np.where(E_flatten!=-1)[0]
        not_empty_num = len(not_empty_ids)
        E_flatten_nonan = E_flatten[not_empty_ids]

        # generate sample weights that balances different lengths
        sample_weights = np.zeros_like(E) + 1/Ts.reshape(-1,1)#
        sample_weights = sample_weights.flatten()[not_empty_ids]
        sample_weights = sample_weights/sample_weights.mean()
        
        ## load model
        
        with open(self.stan_model_path, 'r') as f:
            model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        if stan_model_path is None:
            cache_fn = 'cached-model-%s.pkl'%code_hash
        else:
            cache_fn = 'cached-model-%s-%s.pkl'%(stan_model_path, code_hash)
        try:
            self.stan_model = pickle.load(open(cache_fn, 'rb'))
        except:
            self.stan_model = pystan.StanModel(model_code=model_code)
            with open(cache_fn, 'wb') as f:
                pickle.dump(self.stan_model, f)
        else:
            print("Using cached StanModel")

        ## feed data
        
        data_feed = {'W':self.W,
                     'N':N,
                     'T':T,
                     #'T0':T0,
                     'ND':self.ND,
                     #'NC':Ctr.shape[-1],
                     'not_empty_num':not_empty_num,
                     'not_empty_ids':not_empty_ids+1,  # +1 for stan
                     'sample_weights':sample_weights,
                     'Eobs_flatten_nonan':E_flatten_nonan,
                     'D':D.transpose(1,0,2),  # because matrix[N,ND] D[T];
                     #'C':Ctr,
                     #'p_start':Pobstr[:,:T0],
                     #'A_start':logit(Pobstr[:,:T0]),
                     }

        ## sampling
        if self.random_state is None:
            self.random_state = np.random.randint(10000)
        if save_path is None:
            save_path = 'model_fit2.pkl'
        """
        self.fit_res = self.stan_model.sampling(data=data_feed,
                                       iter=10000, verbose=True,
                                       chains=1, seed=self.random_state)
        #print(self.fit_res.stansummary(pars=['mu_mu','sigma_mu','sigma_sigma','sigma_alpha', 'sigma_b']))

        # save
        with open(save_path, 'wb') as f:
            pickle.dump([self.stan_model, self.fit_res], f)
        """
        with open(save_path, 'rb') as f:
            self.stan_model, self.fit_res = pickle.load(f)
            
        return self
        
    def predict(self, D, training=False, sim=None):
        """
        D: drug concentration, list of arrays, with different lengths. Each element has shape (T,ND)
        training:
        sim:
        """
        
        if sim is None:
            if training:
                df = self.fit_res.to_dataframe(pars=['t0', 'mu', 'alpha', 'sigma', 'b'])
                N = len(D)
                t0 = df[['t0[%d]'%ii for ii in range(1,N+1)]].values.mean(axis=0)
                mu = df[['mu[%d]'%ii for ii in range(1,N+1)]].values.mean(axis=0)
                alpha = df[['alpha[%d]'%ii for ii in range(1,N+1)]].values.mean(axis=0)
                sigma = df[['sigma[%d]'%ii for ii in range(1,N+1)]].values.mean(axis=0)
                b = np.array([df[['b[%d,%d]'%(ii,jj) for ii in range(1,N+1)]].values for jj in range(1,self.ND+1)])
                b = b.mean(axis=1).T
            else:
                raise NotImplementedError
            t0 = [t0]
            mu = [mu]
            alpha = [alpha]
            sigma = [sigma]
            b = [b]
        else:
            raise NotImplementedError
        
        Ps = [] 
        for sim_i in range(len(t0)):
            P = []
            for i in range(len(D)):
                T = len(D[i])
                tt = np.arange(1, T+1)
                tmp = np.log(tt+t0[sim_i][i])-mu[sim_i][i]
                
                A = alpha[sim_i][i] * np.exp(-tmp**2 / (2* (sigma[sim_i][i]**2))) - (D[i] * b[sim_i][i]).sum(axis=1)
            
                P.append( sigmoid(A) )
                
            Ps.append(P)
            
        return Ps

