import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__=='__main__':
    
    master_sheet_path = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019.xlsx'
    master_sheet = pd.read_excel(master_sheet_path, sheet_name='All Corrections Included')
    master_sheet = master_sheet[~pd.isna(master_sheet.MRN)].reset_index(drop=True)
    master_sheet.MRN = master_sheet.MRN.astype(int)
    
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
    patient_mrn_set = [master_sheet.MRN[master_sheet.Index==sid].values[0] for sid in sids]
    
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

    output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh'
    drug_df.to_csv(os.path.join(output_dir, 'drug_data.csv'), index=False)

