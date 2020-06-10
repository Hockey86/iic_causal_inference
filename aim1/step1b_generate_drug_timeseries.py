import datetime
import glob
import os
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm


if __name__=='__main__':
            
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
    output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh'
   
    # master sheet
    
    master_sheet_path = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019.xlsx'
    master_sheet = pd.read_excel(master_sheet_path, sheet_name='All Corrections Included')
    master_sheet = master_sheet[~pd.isna(master_sheet.MRN)].reset_index(drop=True)
    #master_sheet.MRN = master_sheet.MRN.astype(int)
    mrns = [master_sheet.MRN[master_sheet.Index==sid].values[0] for sid in sids]
    
    # drug table
    
    drug_df = pd.read_csv(os.path.join(output_dir, 'drug_data.csv'))
    drug_df.loc[drug_df.drugNameSimplified=='phenytoin', 'drugNameSimplified'] = 'fosphenytoin'
    drug_df.loc[drug_df.drugNameSimplified=='divalproex', 'drugNameSimplified'] = 'valproate'
    drug_df.Admin_Time = pd.to_datetime(drug_df.Admin_Time, format='%Y-%m-%dT%H:%M:%S')
    drug_df.Dose_Unit = drug_df.Dose_Unit.str.upper()
    drug_df.loc[pd.isna(drug_df.Dose_Unit), 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG_PE', 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG PE', 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG PE/KG', 'Dose_Unit'] = 'MG/KG'
    print(set(drug_df.Dose_Unit))
    
    # human label
    human_label_dir = '/home/sunhaoqi/Desktop/IIC_human_labels'
    human_label_paths = glob.glob(os.path.join(human_label_dir, '*.csv'))
    
    # body weight
    
    df_bodyweights = pd.read_excel('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/bodyweights_second_query_Sean.xlsx', sheet_name='Combined Data - Weight Only')
    df_bodyweights = df_bodyweights.rename(columns={'MGH_MRN':'MRN'})
    df_bodyweights = df_bodyweights.dropna().reset_index(drop=True)
    df_bodyweights.MRN = df_bodyweights.MRN.astype(int)
    bodyweights = []
    bodyweight_times = []
    ages = []
    genders = []
    eeg_start_times = []
    for i in range(len(sids)):
        eeg_file_names = os.listdir('/media/mad3/Projects/SAGE_Data/sid%04d/Data'%int(sids[i][3:]))
        eeg_start_time = min([datetime.datetime.strptime('T'.join(x.replace('.mat','').split('_')[1:]), '%Y%m%dT%H%M%S') for x in eeg_file_names])
        eeg_start_times.append(eeg_start_time)
        
        ids = np.where(df_bodyweights.MRN==mrns[i])[0]
        if len(ids)>0:
            # find the weight closest to eeg time
            delta_times = [np.abs((x-eeg_start_time).total_seconds()) for x in df_bodyweights.Date.iloc[ids]]
            ii = np.argmin(delta_times)
            weight = df_bodyweights.Result.iloc[ids[ii]]
            #if 'ounce' in :
            #    bodyweights.append(weight*0.0283495) # convert ounce to kg
            #elif 'pound' in :
            bodyweights.append(weight*0.453592) # convert pound to kg
            bodyweight_times.append(df_bodyweights.Date.iloc[ids[ii]])
        else:
            bodyweights.append(np.nan)
            bodyweight_times.append(np.nan)
        ages.append(master_sheet.Age[master_sheet.MRN==mrns[i]].iloc[0])
        gender = master_sheet.Gender[master_sheet.MRN==mrns[i]].iloc[0]
        genders.append(1 if gender=='M' else 0)
    bodyweights = np.array(bodyweights)
    bodyweights[bodyweights<=20] = np.nan
    ages = np.array(ages)
    genders = np.array(genders)
    
    # impute missing body weight
    nanids = np.isnan(bodyweights)
    notnanids = np.where(~np.isnan(bodyweights))[0]
    for i in np.where(nanids)[0]:
        closeids = (genders[notnanids]==genders[i]) & (np.abs(ages[i] - ages[notnanids])<=10)
        bodyweights[i] = np.median(bodyweights[notnanids[closeids]])
    
    df_bodyweights2 = pd.DataFrame(data=np.c_[sids, bodyweights, bodyweight_times, eeg_start_times, nanids],
                                   columns=['sid', 'body weight(kg)', 'body weight time', 'eeg start time', 'imputed'])
    df_bodyweights2.to_csv(os.path.join(output_dir, 'body_weights.csv'), index=False)
    
    # for each patient
    
    for si, sid in enumerate(tqdm(sids)):
        if os.path.exists(os.path.join(output_dir, sid)):
            continue
        drug_df_ = drug_df[drug_df.MRN==mrns[si]]
        drug_df_ = drug_df_.sort_values('Admin_Time').reset_index(drop=True)
        eeg_start_time = eeg_start_times[si]
        
        humanlabel = pd.read_csv([x for x in human_label_paths if 'sid%04d'%int(sid[3:]) in x][0], header=None)
        #nan_ids = np.isnan(humanlabel[0].values)
        T = len(humanlabel)
            
        # generate a time series based on drug_df_ (no data is 0)
        #drug_ts = {}
        drugnames = set(drug_df_.drugNameSimplified)
        for drugname in drugnames:
            this_drug_df_ = drug_df_[drug_df_.drugNameSimplified==drugname].reset_index(drop=True)
            this_drug_ts_weightnormalized = np.zeros(T)
            this_drug_ts = np.zeros(T)
            for i in range(len(this_drug_df_)):
                if this_drug_df_.Dose_Unit.iloc[i] in ['MG', 'MG/KG', 'MCG']:
                    start = int(round((this_drug_df_.Admin_Time.iloc[i]-eeg_start_time).total_seconds()/2.))
                    end = start+60//2
                    if this_drug_df_.Dose_Unit.iloc[i] == 'MCG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60/1000.
                    elif this_drug_df_.Dose_Unit.iloc[i] == 'MG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60
                    elif this_drug_df_.Dose_Unit.iloc[i] == 'MG/KG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60*bodyweights[si]
                        
                elif this_drug_df_.Dose_Unit.iloc[i] in ['MG/HR', 'MG/KG/HR', 'MCG/KG/HR', 'MCG/KG/MIN']:
                    start = int(round((this_drug_df_.Admin_Time.iloc[i]-eeg_start_time).total_seconds()/2.))
                    if i==len(this_drug_df_)-1:
                        end = len(this_drug_ts)
                    else:
                        end = int(round((this_drug_df_.Admin_Time.iloc[i+1]-eeg_start_time).total_seconds()/2.))
                    if this_drug_df_.Dose_Unit.iloc[i] in ['MG/HR']:
                        dose = this_drug_df_.Dose_Amount.iloc[i]
                    elif this_drug_df_.Dose_Unit.iloc[i] in ['MG/KG/HR']:
                        dose = this_drug_df_.Dose_Amount.iloc[i]*bodyweights[si]
                    elif this_drug_df_.Dose_Unit.iloc[i] in ['MCG/KG/HR']:
                        dose = this_drug_df_.Dose_Amount.iloc[i]/1000.*bodyweights[si]
                    elif this_drug_df_.Dose_Unit.iloc[i] in ['MCG/KG/MIN']:
                        dose = this_drug_df_.Dose_Amount.iloc[i]/1000.*60*bodyweights[si]
                    
                start = max(0,start)
                end = min(len(this_drug_ts),end)
                if end>start:
                    this_drug_ts[start:end] = dose
                    this_drug_ts_weightnormalized[start:end] = dose/bodyweights[si]
            
            # fill nans to gaps
            #this_drug_ts[nan_ids] = np.nan
            
            if not os.path.exists(os.path.join(output_dir, sid)):
                os.mkdir(os.path.join(output_dir, sid))
            """
            # save into csv
            datetimes = [datetime.datetime.strftime(eeg_start_time+datetime.timedelta(seconds=x*2), '%Y/%m/%d %H:%M:%S') for x in range(T)]
            save_df = pd.DataFrame(data={'Date&Time': datetimes, 'Drug Amount':this_drug_ts, 'Drug Amount (Normalized by Body Weight)':this_drug_ts_weightnormalized})
            save_df.to_csv(os.path.join(output_dir, sid, '%s_%s_2secWindow.csv'%(sid, drugname)), index=False)
            """
            # save into mat
            res = {'start_time': datetime.datetime.strftime(eeg_start_time, '%Y/%m/%d %H:%M:%S'),
                   'drug_dose': csr_matrix(this_drug_ts),
                   'drug_dose_bodyweight_normalized': csr_matrix(this_drug_ts_weightnormalized)}
            sio.savemat(os.path.join(output_dir, sid, '%s_%s_2secWindow.mat'%(sid, drugname)), res)
            
