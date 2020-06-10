from datetime import timedelta
from dateutil.parser import parse
import glob
import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.stats import linregress
import pandas as pd
from tqdm import tqdm
import mne

window_time = 10  # [s]
step_time = 2   # [s]
pad_time = 4    # [s]


def get_signal_length(path, return_Fs=False):
    with h5py.File(path, 'r') as ff:
        length = len(ff['data'])
        Fs = ff['Fs'][0,0]
    
    if return_Fs:
        return length/Fs, Fs
    else:
        return length/Fs
        
        
def get_features(path):
    parts = path.split(os.sep)
        
    with h5py.File(path, 'r') as eeg_res:
        eeg = eeg_res['data'][:].T
        Fs = eeg_res['Fs'][0,0]
        #channels = 
        
    # montage
    chs1 = [0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]
    chs2 = [4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]
    eeg = eeg[chs1] - eeg[chs2]
    #channels = ['%s-%s'%(channels[chs1[ii]], channels[chs2[ii]]) for ii in range(len(chs1))]
    
    # filtering
    eeg = mne.filter.notch_filter(eeg, Fs, 60, n_jobs=1, verbose=False)
    eeg = mne.filter.filter_data(eeg, Fs, 0.5, 40, n_jobs=1, verbose=False)
    
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    
    # segment
    start_ids = np.arange(0, eeg.shape[1]-window_size+1, step_size)
    segs = eeg[:, list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)
    #start_time = start_time + datetime.timedelta(seconds=pad_time)
        
    # spec
    """
    spec, spec_freq = mne.time_frequency.psd_array_multitaper(segs, Fs,
                        fmin=0.5, fmax=20, bandwidth=1, n_jobs=4,
                        verbose=False, normalization='full')
    # reduce size
    spec = spec[...,::2]
    spec_freq = spec_freq[::2]
    spec[np.isinf(spec)] = np.nan
    spec_db = np.nanmean(spec, axis=1)
    #plt.imshow(spec.T,aspect='auto',vmin=-10,vmax=20,origin='lower',cmap='jet');plt.show()
    
    # pad spec
    padding = np.zeros((int(round(pad_time/step_time)), spec_db.shape[1]))+np.nan
    spec_db = np.concatenate([padding, spec_db, padding], axis=0)
    """
    spec_path = os.sep.join(parts[:-2] + ['Spectrograms', parts[-1].replace('.mat', '_spect.mat')])
    spec_res = sio.loadmat(spec_path)
    spec_db = []
    for i in range(spec_res['Sdata'].shape[0]):
        spec = spec_res['Sdata'][i,0].astype(float)
        spec = 10*np.log10(spec)
        spec[np.isinf(spec)] = np.nan
        spec_db.append(spec)
    spec_db = np.array(spec_db).transpose(2,0,1)
    spec_db = np.nanmean(spec_db, axis=1)
    spec_freq = spec_res['sfreqs'].flatten()
    
    # spike
    spike_path = os.sep.join(parts[:-2] + ['Features/ssd', parts[-1].replace('.mat', '_ssd.mat')])
    spike_res = sio.loadmat(spike_path)
    spike = spike_res['yp'].flatten()
    artifact = spike_res['artifact'].flatten()
    assert len(spike)==len(artifact)==eeg.shape[1]
    
    spike = (spike>=0.43).astype(float)
    spike[artifact==1] = np.nan
    start_ids = np.arange(0, len(spike)-step_size+1, step_size)
    spike = spike.reshape(1,-1)[:, list(map(lambda x:np.arange(x,x+step_size), start_ids))][0]
    spike = np.nanmax(spike, axis=1)
    
    assert len(spec_db)==len(spike)
    return spec_db, spec_freq, spike
    
    
if __name__=='__main__':

    output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
    
    #eeg_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/raw_eeg_SE_pts'
    #eeg_dir = '/media/mad3/Projects/IIC_Projects/SZ_Cases_52_IIIC_Labeling_Phase1/done'
    eeg_dir = '/media/mad3/Projects/SAGE_Data'
    
    #human_label_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/human_annotation_from_brandon/Label'
    human_label_dir = '/home/sunhaoqi/Desktop/IIC_human_labels'
    human_label_paths = glob.glob(os.path.join(human_label_dir, '*.csv'))
    
    drug_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh'
    #drug_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/SZ Cases 52 IIC labels/Medication/Haoqi/AED_2_Sec_Mat'
    
    sids = ['sid36', 'sid39', 'sid56', 'sid297', 'sid327', 'sid385',
        'sid395', 'sid400', 'sid403', 'sid406', 'sid424', 'sid450',
        'sid456', 'sid490', 'sid512', 'sid551', 'sid557', 'sid575',
        'sid988', 'sid1016', 'sid1025', 'sid1034', 'sid1038', 'sid1039',
        'sid1055', 'sid1056', 'sid1063', 'sid1337', 'sid1897', 'sid1913',
        'sid1915', 'sid1916', 'sid1917', 'sid1926', 'sid1928', 'sid1956',
        'sid1966']+\
       ['sid2', 'sid23', 'sid45', 'sid77', 'sid91', 'sid741', 'sid821', 'sid832', 'sid848',
        'sid8', 'sid24', 'sid54', 'sid82', 'sid92', 'sid771', 'sid822', 'sid833', 'sid849',
        'sid11', 'sid28', 'sid57', 'sid84', 'sid97', 'sid801', 'sid823', 'sid834', 'sid852',
        'sid13', 'sid30', 'sid61', 'sid88', 'sid734', 'sid808', 'sid824', 'sid837', 'sid856',
        'sid17', 'sid38', 'sid69', 'sid89', 'sid736', 'sid815', 'sid827', 'sid839',
        'sid18', 'sid44', 'sid71', 'sid90', 'sid739', 'sid817', 'sid828', 'sid845']+\
        ['sid863', 'sid864', 'sid865', 'sid870', 'sid872', 'sid875', 'sid876', 'sid880',
         'sid881', 'sid884', 'sid886', 'sid887', 'sid890', 'sid914', 'sid915', 'sid917',
         'sid918', 'sid927', 'sid933', 'sid940', 'sid942', 'sid944', 'sid952', 'sid960',
         'sid963', 'sid965', 'sid967', 'sid983', 'sid984', 'sid987', 'sid994', 'sid1000',
         'sid1002', 'sid1006', 'sid1022', 'sid1024', 'sid1101', 'sid1102', 'sid1105',
         'sid1113', 'sid1116']
    
    NSAED_list = ['levetiracetam', 'lacosamide', 'lorazepam', 'phenytoin',
                 'fosphenytoin', 'phenobarbital', 'carbamazepine',
                 'valproate', 'divalproex', 'topiramate', 'clobazam', 'lamotrigine',
                 'oxcarbazepine', 'diazepam', 'zonisamide', 'clonazepam']
    SAED_list = ['propofol', 'midazolam',  'ketamine', 'pentobarbital']
    Dnames = SAED_list + NSAED_list
    
    ## preprocess clinical variables
    for sid in tqdm(sids):
        save_path = os.path.join(output_dir, sid+'.mat')
        if os.path.exists(save_path):
            continue
        res = {}#sio.loadmat(save_path)
        
        ## get human label
        human_label = pd.read_csv([x for x in human_label_paths if 'sid%04d'%int(sid[3:]) in x][0], header=None)[0].values
        res['human_iic'] = human_label
        T = len(human_label)
        
        ## get drugs
        
        drugs = []
        drugs_weightnormalized = []
        for dn in Dnames:
            read_path = os.path.join(drug_dir, '%s/%s_%s_2secWindow.mat'%(sid, sid, dn))
            #read_path = os.path.join(drug_dir, 'sid%04d/sid%04d_%s_2secWindow.mat'%(int(sid[3:]), int(sid[3:]), dn))
            if os.path.exists(read_path):
                drug_res = sio.loadmat(read_path)
                # get 2s widnows of drug doses
                this_drug = drug_res['drug_dose'].toarray().flatten()
                #this_drug = np.array([x[0,0] for x in drug_res['RAW_Buff_inc_header'][1:,1]]).astype(float)
                if T>len(this_drug):
                    this_drug = np.r_[this_drug, np.zeros(T-len(this_drug))]
                else:
                    this_drug = this_drug[:T]
                
                # get 2s widnows of drug doses (normalized by body weight)
                this_drug2 = drug_res['drug_dose_bodyweight_normalized'].toarray().flatten()
                #this_drug2 = np.array([x[0,0] for x in drug_res['RAW_Buff_inc_header'][1:,2]]).astype(float)
                if T>len(this_drug2):
                    this_drug2 = np.r_[this_drug2, np.zeros(T-len(this_drug2))]
                else:
                    this_drug2 = this_drug2[:T]
                    
            else:
                this_drug = np.zeros(T)
                this_drug2 = np.zeros(T)
            drugs.append(this_drug)
            drugs_weightnormalized.append(this_drug2)
        res['Dnames'] = Dnames
        res['drug'] = np.array(drugs).T
        res['drugs_weightnormalized'] = np.array(drugs_weightnormalized).T
        
        ## TODO get predicted IIC
        #res['iic'] = res['iic'][:T]
        
        ## get spec, spike
        eeg_paths = glob.glob(os.path.join(eeg_dir, 'sid%04d'%int(sid[3:]), 'Data', 'sid*.mat'))
            
        # get start and end time
        start_times = [parse(' '.join(os.path.basename(x)[3:-4].split('_')[1:])) for x in eeg_paths]
        start_time = min(start_times)
        max_time = max(start_times)
        seconds = get_signal_length(eeg_paths[start_times.index(max_time)])
        end_time = max_time + timedelta(seconds=seconds)
        
        # creat empty array to contain all data for this subject
        totalT = int(np.ceil((end_time - start_time).total_seconds()/step_time))
        this_subject_spec_db = np.zeros((totalT, 50))+np.nan
        this_subject_spike = np.zeros(totalT)+np.nan
        
        # loop over all recordings of this subject
        for fi, file_path in enumerate(tqdm(eeg_paths)):
            try:
                spec_db, freq, spike = get_features(file_path)#, iic, freq, bsr, spat
            except Exception as ee:
                print('Error: %s\n%s'%(file_path, str(ee)))
                continue
            this_start_time = int(round((start_times[fi] - start_time).total_seconds()/step_time))
            this_subject_spec_db[this_start_time:this_start_time+len(spec_db)] = spec_db
            this_subject_spike[this_start_time:this_start_time+len(spec_db)] = spike
        """
        res2 = sio.loadmat(save_path)
        this_subject_spec_db = res2['spec']
        this_subject_spike = res2['spike'].flatten()
        freq = res2['spec_freq'].flatten()
        """
        T = min(T, len(this_subject_spec_db))
        this_subject_spec_db = this_subject_spec_db[:T]
        this_subject_spike = this_subject_spike[:T]
        res['human_iic'] = res['human_iic'][:T]
        res['drug'] = res['drug'][:T]
        res['drugs_weightnormalized'] = res['drugs_weightnormalized'][:T]
        
        res['spike'] = this_subject_spike
        res['spec'] = this_subject_spec_db
        res['spec_freq'] = freq
        res['start_time'] = start_time.strftime('%Y/%m/%d %H:%M:%S')
    
        ## get artifact indicator
        
        totalpower = np.nansum(np.power(10, this_subject_spec_db/10), axis=1)*(freq[1]-freq[0])
        goodids = np.abs(totalpower)>1e-4
        totalpower = 10*np.log10(totalpower)
        totalpower2 = totalpower[goodids]
        slopes = np.zeros_like(totalpower)+np.nan
        slopes2 = np.array([linregress(freq, this_subject_spec_db[ii]).slope for ii in np.where(goodids)[0]])
        slopes[goodids] = slopes2
        human_label2 = res['human_iic'][goodids]
        iqr = np.nanpercentile(totalpower2,75)-np.nanpercentile(totalpower2,25)
        whis = 3
        tp_lb = np.nanpercentile(totalpower2,25)-whis*iqr
        tp_ub = np.nanpercentile(totalpower2,75)+whis*iqr
        iqr = np.nanpercentile(slopes2,75)-np.nanpercentile(slopes2,25)
        slope_ub = np.nanpercentile(slopes2,75)+whis*iqr
        artifact_indicator = np.zeros_like(totalpower)
        artifact_indicator[(totalpower>tp_ub) | (totalpower<tp_lb) | (slopes>slope_ub) | (~goodids)] = 1
        artifact_ids = np.where(artifact_indicator==1)[0]
        for ii in artifact_ids:
            start = max(0, ii-5)
            end = min(len(artifact_indicator), ii+5)
            artifact_indicator[start:end] = 1
        res['artifact'] = artifact_indicator
        
        # save
        sio.savemat(save_path, res)
        
