import os
import pickle
import sys
import numpy as np
from scipy.special import logit
from joblib import Parallel, delayed
from tqdm import tqdm
from myfunctions import get_pk_k, drug_concentration
SIMULATOR_PATH = 'step6_simulator'
OUTCOME_PREDICTION_PATH = 'step7_outcome_regression'
sys.path.insert(0, SIMULATOR_PATH)
from simulator import *
sys.path.insert(0, OUTCOME_PREDICTION_PATH)
from fit_model_ordinal import generate_outcome_X


def generate_dose(dose_array, interval, lengths):
    res = []
    for i in range(len(lengths)):
        d_ = np.zeros((lengths[i], len(dose_array)))
        ids = np.arange(0, lengths[i], interval)
        d_[ids] = dose_array
        res.append(d_)
    return res


def drug_from_constant(val, drug=None):
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
        return res
    return func


def drug_from_concentration(data):
    def func(Ps, D, C, Dname, sid, t):
        res = data[sid][t]
        return res
    return func


def drug_from_dose(dose_, PK_K):
    concentration = [drug_concentration(dd.T, PK_K).T for dd in dose_]
    def func(Ps, D, C, Dname, sid, t):
        res = concentration[sid][t]
        return res
    return func


def drug_from_policy():
    pass


def evaluate_one_patient(i, D, Dname, Dmax, Pobs, C, cluster, responses, simulator, outcome_model, drug_regime_func, AR_p, W):
    # use simualtor to generate e
    # use drug_regime_func to generate d
    T, ND = D[i].shape
    d = [np.zeros(ND)]*(AR_p-1)
    e = {r:[np.tile(Pobs[r][i][:AR_p], (500,1))] for r in responses}
    A = {}
    #for t in tqdm(range(AR_p, T)):
    for t in range(AR_p, T):
        # Dt = f(E1:t, D1:t-1, C)
        d.append(drug_regime_func([e[r] for r in responses], d, C[i], Dname, i, t))
        
        # Et+1 = g(E1:t, D1:t, C)
        for r in responses:
            if t==AR_p:
                A[r] = logit(np.clip(Pobs[r][i][:AR_p], 1e-6, 1-1e-6)).reshape(1,-1)
            Esim, A[r] = simulator[r].predict(
                    [np.r_[np.array(d)/Dmax, np.zeros((1,ND))]],
                    cluster[[i]], sid_index=[i],
                    Astart=A[r], return_A=True,
                    verbose=False)
            Esim = Esim[0][:,[-1]]
            e[r].append(Esim)
    d.append(np.zeros(ND)+np.nan) # fill the last drug with NaN, so not counted
            
    d = np.array(d)
    e = {r:np.concatenate(e[r], axis=1) for r in responses}

    XC = np.tile(C[i], (500,1))
    
    # predict outcome
    X, Xnames = generate_outcome_X(
            e, np.tile(d,(500,1,1)),
            Dmax, Dname, 'response', responses, W,
            same_length_vectorizable=True)
    X = np.c_[X, XC]
    #Xnames.extend(Cname)
    
    """
    Xsim = np.array([e[r].mean(axis=1) for r in responses]).T
    
    Xdrug = np.array(d)
    Xdrug[Xdrug<1e-6] = np.nan
    Xdrug = np.nanmean(Xdrug, axis=0)
    Xdrug[np.isnan(Xdrug)] = 0
    Xdrug = np.tile(Xdrug, (500,1))
    """

    # create X and y
    #X = np.c_[Xdrug, Xsim, XC]
    yp = outcome_model.predict_proba(X)
    
    return yp, d, e, X
    
    

if __name__=='__main__':
    ## define vars
    
    data_type = 'CNNIIC'
    responses = ['iic_burden_smooth', 'spike_rate']
    simulator_model_type = 'cauchy_expit_lognormal_drugoutside_ARMA'
    outcome_model_type = str(sys.argv[1])
    AR_p = 2
    MA_q = 6
    max_iter = 1000
    Nbt = 0
    n_jobs = 5
    random_state = 2020
    
    responses_txt = '+'.join(responses)
    
    ## load data
    
    with open(f'data_to_fit_{data_type}_{responses_txt}.pickle','rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec(f'{k} = res[\'{k}\']')
    y = Y.astype(int)
    N = len(sids)
    print(f'{N} patients')
    
    ## load simulator
    
    simulator = {}
    for response in responses:
        simulator[response] = Simulator(
            os.path.join(SIMULATOR_PATH, f'model_{simulator_model_type}.stan'),
            W, T0=[AR_p, MA_q], random_state=random_state)
        #model_path = os.path.join(SIMULATOR_PATH, f'results_{response}/model_fit_{data_type}_{response}_{simulator_model_type}{AR_p},{MA_q}_iter{max_iter}.pkl')
        model_path = os.path.join('/data/HaoqiSun', f'model_fit_{data_type}_{response}_{simulator_model_type}{AR_p},{MA_q}_iter{max_iter}.pkl')
        simulator[response].load_model(model_path)
    
    ## load outcome prediction model
    
    with open(os.path.join(OUTCOME_PREDICTION_PATH, f'results_{outcome_model_type}_Nbt{Nbt}_{responses_txt}_response.pickle'), 'rb') as ff:
        res = pickle.load(ff)
    Dmax = res['Dmax']
    outcome_model = res['model']
    
    ## define drug regimes to evaluate
    lengths = [len(x) for x in Ddose]
    keppra_8h_30 = generate_dose([0,30,0,0,0,0,0], 8*3600//600, lengths)
    keppra_8h_60 = generate_dose([0,60,0,0,0,0,0], 8*3600//600, lengths)
    keppra_8h_90 = generate_dose([0,90,0,0,0,0,0], 8*3600//600, lengths)
    PK_K = np.array(get_pk_k()[Dname].values.astype(float))
    drug_regimes = {
        'always_zero':drug_from_constant(0),
        'keppra_8h_30':drug_from_dose(keppra_8h_30, PK_K),
        'keppra_8h_60':drug_from_dose(keppra_8h_60, PK_K),
        'keppra_8h_90':drug_from_dose(keppra_8h_90, PK_K),
        'always_propofol_1':drug_from_constant(1, drug='propofol'),
        'always_propofol_5':drug_from_constant(5, drug='propofol'),
        'always_propofol_10':drug_from_constant(10, drug='propofol'),
        'actual_drug':drug_from_concentration(D),
    }
    #'shuffle_drug':drug_from_dose([d for d in Ddose], PK_K, shuffle=True, random_state=random_state),
    
    ## for each drug regime, evaluate drug regime
    
    Yd = {}
    Ds = {}
    Es = {}
    Xs = {}
    for regime_name, drug_regime_func in drug_regimes.items():
        print(regime_name)
        
        # for each subject
        with Parallel(n_jobs=n_jobs, verbose=False) as par:
            res = par(delayed(evaluate_one_patient)(
                    i, D, Dname, Dmax, Pobs, C, cluster,
                    responses, simulator, outcome_model,
                    drug_regime_func, AR_p, W) for i in tqdm(range(N)))
        Ds[regime_name] = [x[1] for x in res]
        Es[regime_name] = [x[2] for x in res]
        Xs[regime_name] = np.array([x[3] for x in res])
        yd_ = np.array([x[0] for x in res])
        if yd_.shape[-1]==2:
            yd_ = yd_[...,1]  # binary classification
        else:
            yd_ = yd_[...,4:].sum(axis=-1)  # ordinal --> binary
        Yd[regime_name] = yd_#.mean(axis=1)
        mean_ = np.nanmean(np.nanmean(Yd[regime_name], axis=0))
        lb_, ub_ = np.nanpercentile(np.nanmean(Yd[regime_name], axis=0), (2.5,97.5))
        print(f'Y({regime_name}) = {mean_:.4f} [{lb_:.4f} -- {ub_:.4f}]')

        with open(f'res_evaluate_Yd_{outcome_model_type}_{simulator_model_type}{AR_p},{MA_q}.pickle', 'wb') as ff:
            pickle.dump([Yd, Ds, Es, Xs], ff)
    import pdb;pdb.set_trace()
        
"""
ids2=(C[:,1]<=55)&(C[:,-5]<=8.5)&(C[:,-4]<=7)&(C[:,4]==0)&(C[:,15]==0)&(C[:,20]==0)
ids2.sum()
{k:Yd[k][ids2].mean() for k in Yd}
"""
