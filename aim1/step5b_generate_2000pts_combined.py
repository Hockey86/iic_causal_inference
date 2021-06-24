import os
import pickle
import numpy as np


def read_data(folder, data_type, responses):
    """
    able to read and combine from different responses
    """
    res = {}
    sids = set()  # common_sids
    for r in responses:
        with open(os.path.join(folder, f'data_to_fit_{data_type}_{r}.pickle'), 'rb') as f:
            res[r] = pickle.load(f)
        if len(sids)==0:
            sids.update(res[r]['sids'])
        else:
            sids &= set(res[r]['sids'])
    sids = np.array(sorted(sids, key=lambda x:int(x[len('sid'):])))
    
    Pobs = {}
    window_start_ids = {}
    data = {}
    for k in ['W', 'Dname', 'Cname', 'Yname']:
        data[k] = res[responses[0]][k]
    for ri, r in enumerate(responses):
        ids = [res[r]['sids'].index(sid) for sid in sids]
        for k in res[r]:
            if ri==0 and k in ['D','Ddose']:#'window_start_ids'
                data[k] = [res[r][k][x] for x in ids]
            elif ri==0 and k in ['cluster', 'pseudoMRNs', 'C', 'Y']:
                data[k] = np.array(res[r][k])[ids]
            elif k=='Pobs':
                Pobs[r] = [res[r][k][x] for x in ids]
            elif k=='window_start_ids':
                window_start_ids[r] = [res[r][k][x] for x in ids]
    
    # make D, Pobs have same length from different responses according to window_start_ids
    for i, sid in enumerate(sids):
        common_time_steps = sorted(set.intersection(*map(set, [window_start_ids[r][i] for r in responses])))
        for ri, r in enumerate(responses):
            if len(common_time_steps)==len(window_start_ids[r][i]):
                continue
            ids = np.in1d(window_start_ids[r][i], common_time_steps)
            window_start_ids[r][i] = window_start_ids[r][i][ids]
            Pobs[r][i] = Pobs[r][i][ids]
            if ri==0:
                data['D'][i] = data['D'][i][ids]
                data['Ddose'][i] = data['Ddose'][i][ids]
    
    # MAP = 1/3 SBP + 2/3 DBP
    # The sixth report of the Joint National Committee on prevention, detection, evaluation, and treatment of high blood pressure. [Arch Intern Med. 1997]
    C = data['C']
    Cname = data['Cname']
    """
    MAP = C[:,Cname.index('systolic BP')]/3+C[:,Cname.index('diastolic BP')]/3*2
    C = np.c_[C, MAP]
    Cname.append('mean arterial pressure')
    
    remove_names = [
        #'SID', 'cluster',
        'iGCS = T?', 'iGCS-E', 'iGCS-V', 'iGCS-M', 'Worst GCS Intubation status', 'iGCS actual scores', 'APACHE II  first 24',
        'Worst GCS in 1st 24',
        'systolic BP', 'diastolic BP',]
    C = C[:,~np.in1d(Cname, remove_names)]
    for x in remove_names:
        Cname.remove(x)
    """
    
    # remove patients with missing outcome
    ids = np.where(~np.isnan(data['Y']))[0]
    sids = sids[ids]
    pseudoMRNs = data['pseudoMRNs'][ids]
    Pobs = {r:[Pobs[r][x] for x in ids] for r in Pobs}
    D = [data['D'][x] for x in ids]
    Ddose = [data['Ddose'][x] for x in ids]
    C = C[ids]
    Y = data['Y'][ids]
    window_start_ids = [window_start_ids[responses[0]][x] for x in ids]
    cluster = data['cluster'][ids]
    return sids, pseudoMRNs, Pobs,\
           D, Ddose, data['Dname'],\
           C, Cname, Y, data['Yname'],\
           window_start_ids, cluster, data['W']
           
           
if __name__=='__main__':
    
    data_type = 'CNNIIC'
    responses = ['iic_burden_smooth', 'spike_rate']
    
    sids, pmrns, Pobs, D, Ddose, Dname, C, Cname, Y, Yname, window_start_ids, cluster, W = read_data('.', data_type, responses)
    
    responses2 = '+'.join(responses)
    output_path = f'data_to_fit_CNNIIC_{responses2}.pickle'
    print(f'{len(sids)} patients')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'W':W, 'window_start_ids':window_start_ids,
            'D':D, 'Ddose':Ddose, 'Dname':Dname,
            'Pobs':Pobs, 'Pname':responses,
            'C':C, 'Cname':Cname, 'Y':Y, 'Yname':Yname,
            'cluster':cluster, 'sids':sids, 'pseudoMRNs':pmrns,}, f)
