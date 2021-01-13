import os
import pickle
import sys
import numpy as np
from scipy.special import logit
from tqdm import tqdm
SIMULATOR_PATH = 'step6_simulator'
OUTCOME_PREDICTION_PATH = 'step7_outcome_regression'
sys.path.insert(0, SIMULATOR_PATH)
sys.path.insert(0, OUTCOME_PREDICTION_PATH)
from simulator import *
from fit_model import read_data


def drug_from_constant(val, Dmax=None, drug=None):
    def func(Ps, D, C, Dname, sid, t):
        """
        Ps: responses, [array.shape=(T+1,), ...]
        D:  drug, array.shape=(T,)
        C:  covariates, array.shape=(#C,)
        sid: subject index
        t:  current time step
        """
        res = np.zeros(len(Dname))
        if drug is None:  # all drugs
            res = res + val
        else:
            if type(drug)==str:
                res[Dname.index(drug)] = val
            else:
                for d in drug:
                    res[Dname.index(d)] = val
        if Dmax is not None:
            res /= Dmax
        return res
    return func


def drug_from_data(data, Dmax=None):
    def func(Ps, D, C, Dname, sid, t):
        res = data[sid][t]
        if Dmax is not None:
            res /= Dmax
        return res
    return func


#def drug_from_policy():


if __name__=='__main__':
    ## define vars
    
    data_type = 'CNNIIC'
    responses = ['iic_burden_smooth', 'spike_rate']
    simulator_model_type = 'cauchy_expit_lognormal_drugoutside_ARMA'
    outcome_model_type = 'ltr'
    AR_p = 2
    MA_q = 6
    max_iter = 1000
    Nbt = 0
    random_state = 2020
    
    responses_txt = '_'.join(responses)
    
    ## load data
    
    sids, pseudoMRNs, Pobs, D, Dname, C, Cname, Y, Yname, window_start_ids, cluster, W = read_data('.', data_type, responses)
    N = len(sids)
    ND = len(Dname)
    
    ## load simulator
    
    simulator = {}
    for response in responses:
        simulator[response] = Simulator(
            os.path.join(SIMULATOR_PATH, f'model_{simulator_model_type}.stan'),
            W, T0=[AR_p, MA_q], random_state=random_state)
        #model_path = os.path.join(SIMULATOR_PATH, f'results_{response}/model_fit_{data_type}_{response}_{simulator_model_type}{AR_p},{MA_q}_iter{max_iter}.pkl')
        model_path = os.path.join('/data/IIC-Causality', f'model_fit_{data_type}_{response}_{simulator_model_type}{AR_p},{MA_q}_iter{max_iter}.pkl')
        simulator[response].load_model(model_path)
    
    ## load outcome prediction model
    
    with open(os.path.join(OUTCOME_PREDICTION_PATH, f'results_{outcome_model_type}_Nbt{Nbt}_{responses_txt}_response.pickle'), 'rb') as ff:
        res = pickle.load(ff)
    Dmax = res['Dmax']
    outcome_model = res['model']
    
    ## define drug regimes to evaluate
    drug_regimes = {
        'always_zero':drug_from_constant(0, Dmax=Dmax),
        'always_propofol_0.01':drug_from_constant(1, Dmax=Dmax, drug='propofol'),
        'actual_drug':drug_from_data(D, Dmax=Dmax),
    }
    
    ## for each drug regime, evaluate drug regime
    
    Yd = {}
    for regime_name, drug_regime_func in drug_regimes.items():
        print(regime_name)
        
        # for each subject
        Yd[regime_name] = []
        A = {}
        for i in tqdm(range(N)):
            # use simualtor to generate e
            # use drug_regime_func to generate d
            T = len(D[i])
            d = [np.zeros(ND)]*(AR_p-1)
            e = {r:list(Pobs[r][i][:AR_p]) for r in responses}
            for t in range(AR_p, T):#tqdm()
                # Dt = f(E1:t, D1:t-1, C)
                d.append(drug_regime_func([e[r] for r in responses], d, C[i], Dname, i, t))
                
                # Et+1 = g(E1:t, D1:t, C)
                for r in responses:
                    if t==AR_p:
                        A[r] = logit(np.clip(Pobs[r][i][:AR_p], 1e-6, 1-1e-6)).reshape(1,-1)
                    Esim, A[r] = simulator[r].predict(
                            [np.r_[np.array(d), np.zeros((1,ND))]],
                            cluster[[i]], sid_index=[i],
                            Astart=A[r], return_A=True,
                            verbose=False)
                    Esim = Esim[0][:,-1].mean(axis=0)  # averaging simulations
                    e[r].append(Esim)
            d.append(np.zeros(ND)+np.nan) # fill the last drug with NaN, so not counted
                    
            d = np.array(d)
            e = {r:np.array(e[r]) for r in responses}
            
            # predict outcome
            
            Xsim = [e[r].mean() for r in responses]
            #sim_name = [r+'_mean' for r in responses]
            
            Xdrug = np.array(d)
            Xdrug[Xdrug<1e-6] = np.nan
            Xdrug = np.nanmean(Xdrug, axis=0)
            Xdrug[np.isnan(Xdrug)] = 0
            #Dname2 = ['mean_positive_dose_'+x for x in Dname]
    
            # create X and y
            X = np.r_[Xdrug, Xsim, C[i]]
            #Xnames = Dname2 + sim_name + Cname
            yp = outcome_model.predict(X.reshape(1,-1))[0]
            
            Yd[regime_name].append(yp)
            
        print(f'Y({regime_name}) = {np.mean(Yd[regime_name])}')
    import pdb;pdb.set_trace()
        
