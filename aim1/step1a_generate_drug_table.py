from dateutil.parser import parse
import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__=='__main__':
    
    # read master sheet
    master_list = pd.read_excel('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019.xlsx', sheet_name='All Corrections Included')
    # there are errors, fix
    master_list.loc[(master_list.Index=='sid1242')&(master_list['Date of admission ']==parse('10/23/2014')), 'The start day of first EEG'] = parse('10/24/2014 16:46')
    master_list = master_list[master_list.Index!='sid522'].reset_index(drop=True)
    master_list.loc[master_list.Index=='sid1044', 'dateCMO/W-LST, in DC, if not search in QPID (CMO)                no CMO=0'] = parse('1/24/2017')

    # remove duplicate patients by taking earlies admission
    mrns = sorted(set(master_list.MRN))
    ids = []
    for mrn in mrns:
        id_ = np.where(master_list.MRN==mrn)[0]
        if len(set(master_list.Index.iloc[id_]))==len(id_):
            ids.extend(id_)
        else:
            dates = master_list['Date of admission '].iloc[id_].values
            ids.append(id_[np.argmin(dates)])
    print(f'Removing {len(master_list)-len(ids)} duplicate patients by taking the earliest admission')
    print(f'Before: row={len(master_list)}; After: row={len(ids)}')
    master_list = master_list.iloc[ids].reset_index(drop=True)
    master_list['The start day of first EEG'] = pd.to_datetime(master_list['The start day of first EEG'])
    master_list['Index2'] = [int(x[len('sid'):]) for x in master_list.Index]
    master_list = master_list.sort_values('Index2').reset_index(drop=True)
    assert len(set(master_list.Index))==len(master_list)
    #assert len(set(master_list.MRN))==len(master_list)
    master_list.to_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019_HaoqiCorrected.csv', index=False)
    
    """
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
    patient_mrn_set = [master_list.MRN[master_list.Index==sid].values[0] for sid in sids]
    """
    patient_mrn_set = list(master_list.MRN)
    
    NSAED_list = ['levetiracetam', 'lacosamide', 'lorazepam', 'phenytoin',
                 'fosphenytoin', 'phenobarbital', 'carbamazepine',
                 'valproate', 'divalproex', 'topiramate', 'clobazam', 'lamotrigine',
                 'oxcarbazepine', 'diazepam', 'zonisamide', 'clonazepam']
    SAED_list = ['propofol', 'midazolam',  'ketamine', 'pentobarbital']
    drug_list = np.array(SAED_list + NSAED_list)
    
    
    ## process pre epic data
    
    raw_drug_dir = '/media/mad3/Projects/EMAR_MedDataBeforeEPIC/EMAR_DATA_PROCESSED_Excel/eMAR_Processed'
    raw_drug_paths = glob.glob(os.path.join(raw_drug_dir, '*.xlsx'))
    
    # for each drug file, take the subset belong to these patients
    drug_df = []
    for raw_drug_path in tqdm(raw_drug_paths):
        try:
            drug_df_ = pd.read_excel(raw_drug_path)
            drug_df_.MRN = drug_df_.MRN.astype(int)
            ids = np.in1d(drug_df_.MRN, patient_mrn_set) & np.in1d(drug_df_.drugNameSimplified, drug_list)
            if ids.sum()==0:
                continue
            drug_df_ = drug_df_[ids]
        except Exception as ee:
            print(raw_drug_path, str(ee))
            continue
        drug_df.append(drug_df_)
    drug_df = pd.concat(drug_df, axis=0).reset_index(drop=True)
    
    #drug_df.to_excel('drug_data.xlsx', index=False)
    drug_df = drug_df.assign(Admin_Time = pd.to_datetime(drug_df.Date.astype(str)+drug_df.Time.astype(str), format='%Y-%m-%d%H:%M:%S')) 
    
    print(len(set(drug_df.MRN)))
    drug_df = drug_df.assign(BeforeEpic=np.ones(len(drug_df)))
    drug_df = drug_df[['MRN', 'Admin_Time', 'Route', 'Admin_Status', 'Dose_Amount', 'Dose_Unit', 'drugNameSimplified', 'BeforeEpic']]
    
    
    ## process post epic data
    
    drug_df2 = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/all-meds-from-list-20180118.csv')#, nrows=10000)
    drug_df2 = drug_df2[np.in1d(drug_df2.MRN, patient_mrn_set)].reset_index(drop=True)

    drug_df2 = drug_df2[(drug_df2.Value!='-')&(drug_df2.Unit!='-')].reset_index(drop=True)
    
    #drug_list = np.unique(pd.read_csv('/data/IIC-Causality/medications/tostudy_medication_names.csv', sep=',').MedName.str.lower()).astype(str)
    drug_list_re = [re.compile(r'\b%s\b'%x) for x in drug_list]
    unique_meds = np.unique(drug_df2.MedicationName.str.lower()).astype(str)
    matched_mask = np.array([[bool(x.search(med)) for x in drug_list_re] for med in unique_meds])
    assert np.all(np.sum(matched_mask, axis=1)<=1)

    med2simple = {med: drug_list[matched_mask[i]][0] for i, med in enumerate(unique_meds) if matched_mask[i].sum()>0}
    # add a column: SimpleMedName
    drug_df2 = drug_df2.assign(drugNameSimplified=[med2simple.get(x.lower(), np.nan) for x in drug_df2.MedicationName])
    drug_df2 = drug_df2[~pd.isna(drug_df2.drugNameSimplified)].reset_index(drop=True)
    drug_df2 = drug_df2.assign(BeforeEpic=np.zeros(len(drug_df2)))
    
    drug_df2 = drug_df2[['MRN', 'AdministrationInstant', 'AdminRoute', 'Action', 'Value', 'Unit', 'drugNameSimplified', 'BeforeEpic']]
    drug_df2 = drug_df2.rename(columns={'AdministrationInstant':'Admin_Time',
                                       'AdminRoute':'Route',
                                       'Action':'Admin_Status',
                                       'Value':'Dose_Amount',
                                       'Unit':'Dose_Unit',})
    drug_df = pd.concat([drug_df, drug_df2], axis=0)

    #output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh'
    output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data'
    import pdb;pdb.set_trace()
    drug_df.to_csv(os.path.join(output_dir, 'drug_data_2000pts.csv'), index=False)

