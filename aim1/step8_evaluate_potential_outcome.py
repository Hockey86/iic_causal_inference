import os
import pickle
import sys
import numpy as np
from tqdm import tqdm
SIMULATOR_PATH = 'step6_simulator'
OUTCOME_PREDICTION_PATH = 'step7_outcome_regression'
sys.path.insert(0, SIMULATOR_PATH)
sys.path.insert(0, OUTCOME_PREDICTION_PATH)
from simulator import *
from fit_model import read_data


def always_constant(val, drug=None):
    def func(P, D, C, Dname):
        """
        P: response, array.shape=(T+1,)
        D: drug, array.shape=(T,)
        C: covariates, array.shape=(#C,)
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
        return res
    return func


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
    
    import pdb;pdb.set_trace()
    sids, pseudoMRNs, Pobs, D, Dname, C, Cname, Y, Yname, window_start_ids, cluster, W = read_data('.', data_type, responses)
    N = len(sids)
    
    ## load simulator
    
    simulator = {}
    for response in responses:
        simulator[response] = Simulator(
            os.path.join(SIMULATOR_PATH, f'model_{simulator_model_type}.stan'),
            W, T0=[AR_p, MA_q], random_state=random_state)
        model_path = os.path.join(SIMULATOR_PATH, f'results_{response}/model_fit_{data_type}_{response}_{simulator_model_type}{AR_p},{MA_q}_iter{max_iter}.pkl')
        simulator[response].load_model(model_path)
    
    ## load outcome prediction model
    
    with open(os.path.join(OUTCOME_PREDICTION_PATH, f'results_{outcome_model_type}_Nbt{Nbt}_{responses_txt}_response.pickle'), 'rb') as ff:
        outcome_model = ff['model']
        Dmax = ff['Dmax']
    
    ## define drug regimes to evaluate
    drug_regimes = {
        'always_zero':always_constant(0),
        'always_propofol_one':always_constant(1, drug='propofol'),
    }
    
    ## for each drug regime, evaluate drug regime
    
    Yd = {}
    for regime_name, drug_regime_func in drug_regimes.items():
        print(regime_name)
        
        # for each subject
        Yd[regime_name] = []
        for i in tqdm(range(N)):
            # use simualtor to generate e
            # use drug_regime_func to generate d
            T = len(D[i])
            d = []
            e = {}
            for response in responses:
                e[response] = list(Pobs[response][i][:AR_p])
                for t in range(AR_p, T):
                    d.append(drug_regime_func(e[response], d, C[i], Dname))
                    Esim = simulator[response].predict(
                            [d], cluster[[i]], sid_index=[i],
                            Pstart=Pobs[response][i][:AR_p].reshape(1,-1))
                    Esim = Esim[0][:,-1].mean(axis=0)  # averaging simulations
                    e[response].append(Esim)

            # predict outcome
            
            Xsim = [e[response].mean() for x in responses]
            sim_name = [x+'_mean' for x in responses]
            
            d = np.array(d)
            Xdrug = np.array(d)
            Xdrug[Xdrug<1e-6] = np.nan
            Xdrug = np.nanmean(Xdrug, axis=0)
            Xdrug[np.isnan(Xdrug)] = 0
            Xdrug = Xdrug/Dmax
            Dname = ['mean_positive_dose_'+x for x in Dname]
    
            # create X and y
            X = np.r_[Xdrug, Xsim, C[i]]
            Xnames = Dname + sim_name + Cname
            yp = outcome_model.predict(X.reshape(1,-1))[0]
            
            Yd[regime_name].append(yp)
            
        print(f'Y({regime_name}) = {np.mean(Yd[regime_name])}')
        
