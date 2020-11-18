import os
import numpy as np
import pandas as pd


def convert_to_onehot(A, id2diag):
    A_onehot = {x:[0]*len(A) for x in id2diag.values()}
    for i, dd in enumerate(A):
        if type(dd)==str:
            dd = dd.replace('.', ',')
            if ',' in dd:
                this_row_diags = [int(x) for x in dd.strip().split(',')]
            else:
                this_row_diags = [int(x) for x in dd.strip().split(' ')]
            for trd in this_row_diags:
                A_onehot[id2diag[trd]][i] = 1
    return A_onehot
    
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
"""
#output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'
       
master_list = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019_HaoqiCorrected.csv')

all_covs = master_list.copy()
all_covs.columns = all_covs.columns.str.replace('[nN]o\s*=\s*0\s*,*\s* [yY]es\s*=\s*1','').str.strip()
all_covs = all_covs.rename(columns={'Final Neuro  Dx    ischemic stroke =1  hemorrhagic stroke=2  SAH=3  SDH=4   SDH+other TBI (including SAH)=5 Other TBI (including SAH)=6  Seizures/status epilepticus=7  Brain tumor/primary or mets=8  CNS infection/inflammation=9   Hypoxic ischemic encephalopathy/Anoxic brain injury=10   Toxic metabolic encephalopathy=11  Other neurosurgical (non tumor/trauma) (eg EVD/VPS etc)=12  Primary psychiatric disorder=13  Other structural-degenerative diseases=18  Spells=19':'Final Neuro  Dx',
'Primary systemic Dx  none=0  Respiratory disorders=1 cardiovascular disorders=2 Renal failure=3 Liver disorders=4 Gastrointestinal non-hemorrhagic=5 Genitourinary=6 Systemic hemorrhage=7 Endocrine emergency=8 Non head trauma (including spine)=9 Malignancy (solid tumors/hematologic)=10 Other post operative (eg CT surgery, transplant)=11 Primary hematological disorder=12  Immunological=13 Dermatological/Musculoskeletal=14':'Primary systemic Dx',
'Sz Semiology,none 0,  generalized=1 focal motor/simple partial/ complex partial = 2':'Sz Semiology1',
'marrital status  unmarried (if the pt was divorced or is widow is still in this category)=0 married=1':'marrital',
'iGCS                          actual scores':'iGCS actual scores'})
# convert midline shift to 0/1
midline = np.zeros(len(all_covs)).astype(int)
midline[~pd.isna(all_covs['Midline shift with any reason ( Document Date)'])] = 1
all_covs['Midline shift with any reason ( Document Date)'] = midline

# convert M to 1, F to 0
sex = np.zeros(len(all_covs)).astype(int)
sex[all_covs['Gender']=='M'] = 1
all_covs['Gender'] = sex

# convert neuro diagnosis to one-hot
neuro_id2diag = {0:'neuro_dx_none', 1:'neuro_dx_ischemic stroke', 2:'neuro_dx_hemorrhagic stroke', 3:'neuro_dx_SAH', 4:'neuro_dx_SDH', 5:'neuro_dx_SDH+other TBI (including SAH)', 6:'neuro_dx_Other TBI (including SAH)', 7:'neuro_dx_Seizures/status epilepticus', 8:'neuro_dx_Brain tumor/primary or mets', 9:'neuro_dx_CNS infection/inflammation', 10:'neuro_dx_Hypoxic ischemic encephalopathy/Anoxic brain injury', 11:'neuro_dx_Toxic metabolic encephalopathy', 12:'neuro_dx_Other neurosurgical (non tumor/trauma) (eg EVD/VPS etc)', 13:'neuro_dx_Primary psychiatric disorder', 18:'neuro_dx_Other structural-degenerative diseases', 19:'neuro_dx_Spells'}
neuro_diagnosis_onehot = convert_to_onehot(all_covs['Final Neuro  Dx'], neuro_id2diag)

# convert primary systemic diagnosis to one-hot
prim_id2diag = {0:'prim_dx_none', 1:'prim_dx_Respiratory disorders', 2:'prim_dx_cardiovascular disorders', 3:'prim_dx_Renal failure', 4:'prim_dx_Liver disorders', 5:'prim_dx_Gastrointestinal non-hemorrhagic', 6:'prim_dx_Genitourinary', 7:'prim_dx_Systemic hemorrhage', 8:'prim_dx_Endocrine emergency', 9:'prim_dx_Non head trauma (including spine)', 10:'prim_dx_Malignancy (solid tumors/hematologic)', 11:'prim_dx_Other post operative (eg CT surgery, transplant)', 12:'prim_dx_Primary hematological disorder', 13:'prim_dx_Immunological', 14:'prim_dx_Dermatological/Musculoskeletal'}
prim_diagnosis_onehot = convert_to_onehot(all_covs['Primary systemic Dx'], prim_id2diag)

# convert Sz semiology to one-hot
sz_id2diag = {0:'sz_dx_none', 1:'sz_dx_generalized', 2:'sz_dx_focal motor/simple partial/ complex partial'}
sz1_diagnosis_onehot = convert_to_onehot(all_covs['Sz Semiology1'], sz_id2diag)
#'Sz Semiology2'
#sz2_diagnosis_onehot = convert_to_onehot(all_covs['Sz Semiology2'], sz_id2diag)

diagnosis_onehot = neuro_diagnosis_onehot.copy()
diagnosis_onehot.update(prim_diagnosis_onehot)
diagnosis_onehot.update(sz1_diagnosis_onehot)
#diagnosis_onehot.update(sz2_diagnosis_onehot)
all_covs = all_covs.assign(**diagnosis_onehot)
all_covs = all_covs.drop(columns=['Final Neuro  Dx', 'Primary systemic Dx'])

#covs = covs.assign(Weight=weights)

cov_names = [
'Gender',
'Age',
'marrital',
#'Weight',

'APACHE II  first 24',
#'temp/F highest (first 24h)', 'temp/F lowest (first 24h)',
#'SBP Highest (first 24h)', 'SBP Lowest (first 24h)',
#'DBP Highest (first 24h)', 'DBP lowest (first 24h)',
#'HR highest (first 24h)', 'HR lowest (first 24h)',
#'RR highest (first 24h)', 'RR lowest (first 24h)',

'Hx CVA (including TIA)',
'Hx HTN',
'Hx Sz /epilepsy',
'Hx brain surgery',
'Hx CKD',
'Hx CAD/MI',
'Hx CHF',
'Hx DM',
'Hx of HLD',
'Hx PUD',
'Hx liver failure',
'Hx tobacco (including ex-smokers)',
'Hx ETOH abuse any time in their life (just when in the hx is mentioned)',
'Hx other substance abuse, any time in their life',
'Hx cancer (other than CNS cancer)',
'Hx CNS cancer',
'Hx PVD',
'Hx dementia',
'Hx COPD/ Asthma',
'Hx leukemia/lymphoma',
'Hx AIDs',
'Hx CTD',
'premorbid MRS before admission  (modified ranking scale),before admission',

#'OSH time 1st AED     (Sz med), just MGH notes, do not look for OSH notes',
#'OSH other Rx, No  0   immunosuppressive 1 Other seizure Meds=2 Both immunosupp and other seizure med =3',
#'CA (PEA) presentation on admission(document Date)',

'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)',
#'Sz Semiology1',
#'Sz Semiology2',
#'Electrographic seizures',  # no data

'elevated ICP=more than 20 (either on admission or in hospital course)   QPID',
'hydrocephalus  (either on admission or during hospital course)   QPID',

'iMV  (initial (on admission) mechanical ventilation)',
'systolic BP',
'diastolic BP',

'Midline shift with any reason ( Document Date)',

# hospital acquired infections are obtained after admission
#'Other HAI',
#'HAI-PNA',
#'HAI-UTI',
#'HAI-Sepsis/SEPTICEMIA',
#'HAI-meningitis/ventriculitis/cerebritis',
#'HAI-Cdiff',
#'DVT',

#'Final Neuro  Dx',
#'Primary systemic Dx',

'Primary systemic dx Sepsis/Shock',

'External Ventricular Drain (EVD)',
'BOLT N0=0 Yes=1',

'iGCS-Total',
'iGCS = T?',
'iGCS-E',
'iGCS-V',
'iGCS-M',
'Worst GCS in 1st 24',
'Worst GCS Intubation status',
'iGCS actual scores',
]+list(neuro_id2diag.values())+list(prim_id2diag.values())+list(sz_id2diag.values())

all_covs_sids = list(all_covs.Index)
covs = all_covs[['Index']+cov_names]
#ids = [all_covs_sids.index(sid) for sid in sids]
#covs = covs.iloc[ids].reset_index(drop=True)

# remove rare covs
notrare_cov_col_ids = (covs.values!=0).sum(axis=0)>=10
covs = covs[covs.columns[notrare_cov_col_ids]]

cols = ['Index', 'Gender', 'Age', 'marrital', 'APACHE II  first 24', 'Hx CVA (including TIA)', 'Hx HTN', 'Hx Sz /epilepsy', 'Hx brain surgery', 'Hx CKD', 'Hx CAD/MI', 'Hx CHF', 'Hx DM', 'Hx of HLD', 'Hx tobacco (including ex-smokers)', 'Hx ETOH abuse any time in their life (just when in the hx is mentioned)', 'Hx other substance abuse, any time in their life', 'Hx cancer (other than CNS cancer)', 'Hx CNS cancer', 'Hx COPD/ Asthma', 'premorbid MRS before admission  (modified ranking scale),before admission', 'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)', 'hydrocephalus  (either on admission or during hospital course)   QPID', 'iMV  (initial (on admission) mechanical ventilation)', 'systolic BP', 'diastolic BP', 'Midline shift with any reason ( Document Date)', 'Primary systemic dx Sepsis/Shock', 'iGCS-Total', 'iGCS = T?', 'iGCS-E', 'iGCS-V', 'iGCS-M', 'Worst GCS in 1st 24', 'Worst GCS Intubation status', 'iGCS actual scores', 'neuro_dx_Seizures/status epilepticus', 'prim_dx_Respiratory disorders']
covs = covs[cols]
covs.to_csv(os.path.join(output_dir, 'covariates.csv'), index=False)


outcome_names = [
'DC MRS (modified ranking scale)',
'DC GOSE (extended glasgow outcome scale)',
'DC dispo home=1, rehab=2, SNF =3, hospice =4, dead =5'
]
outcomes = all_covs[['Index']+outcome_names]
#outcomes = outcomes.iloc[ids].reset_index(drop=True)
outcomes.to_csv(os.path.join(output_dir, 'outcomes.csv'), index=False)
