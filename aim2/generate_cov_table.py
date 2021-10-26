import pickle
import numpy as np
import pandas as pd

cov_mappings = [
['Age', 'Age, year', 'cont'],
['Gender', 'Male gender', 'bin'],
# Race
['Race', 'Asian', 'bin'],
['Race', 'Black / African American', 'bin'],
['Race', 'White / Caucasian', 'bin'],
['Race', 'Other', 'bin'],
['Race', 'Unavailable / Declined', 'bin'],
['Marrital', 'Married', 'bin'],
['premorbid MRS', 'Premorbid mRS before admission', 'cont'],
['APACHE II 1st 24h', 'APACHE II in first 24h', 'cont'],

['iGCS-Total', 'Initial GCS', 'cont'],
['iGCS = T?', 'Initial GCS is with intubation', 'bin'],
['Worst GCS in 1st 24h', 'Worst GCS in first 24h', 'cont'],
['Worst GCS intub', 'Worst GCS in first 24h is with intubation', 'bin'],
['Surgery', 'Admitted due to surgery', 'bin'],
['CA (PEA)', 'Cardiac arrest at admission', 'bin'],
['Sz at presentation', 'Seizure at presentation', 'bin'],
['acute SDH', 'Acute SDH at admission', 'bin'],
['OSH time 1st AED', 'Take anti-epileptic drugs outside hospital', 'bin'],
['HR highest (1std 24h)', 'Highest heart rate in first 24h, /min', 'cont'],
['HR lowest (1st 24h)', 'Lowest heart rate in first 24h, /min', 'cont'],
['SBP highest (1st 24h)', 'Highest systolic BP in first 24h, mmHg', 'cont'],
['SBP lowest (1st 24h)', 'Lowest systolic BP in first 24h, mmHg', 'cont'],
['DBP highest (1st 24h)', 'Highest diastolic BP in first 24h, mmHg', 'cont'],
['DBP lowest (1st 24h)', 'Lowest diastolic BP in first 24h, mmHg', 'cont'],
['EEG day1 MV', 'Mechanical ventilation on the first day of EEG', 'bin'],
['EEG day1 sysBP', 'Systolic BP on the first day of EEG, mmHg', 'cont'],
['EEG day1 GCS', 'GCS on the first day of EEG', 'cont'],

# History of
['Hx CVA', 'Stroke', 'bin'],
['Hx HTN', 'Hypertension', 'bin'],
['Hx Sz', 'Seizure or epilepsy', 'bin'],
['Hx brain surgery', 'Brain surgery', 'bin'],
['Hx CKD', 'Chronic kidney disorder', 'bin'],
['Hx CAD/MI', 'Coronary artery disease and myocardial infarction', 'bin'],
['Hx CHF', 'Congestive heart failure', 'bin'],
['Hx DM', 'Diabetes mellitus', 'bin'],
['Hx HLD', 'Hypersensitivity lung disease', 'bin'],
['Hx PUD', 'Peptic ulcer disease', 'bin'],
['Hx liver failure', 'Liver failure', 'bin'],
['Hx smoking', 'Smoking', 'bin'],
['Hx alcohol', 'Alcohol abuse', 'bin'],
['Hx substance abuse', 'Substance abuse', 'bin'],
['Hx cancer', 'Cancer (except central nervous system)', 'bin'],
['Hx CNS cancer', 'Central nervous system cancer', 'bin'],
['Hx PVD', 'Peripheral vascular disease', 'bin'],
['Hx dementia', 'Dementia', 'bin'],
['Hx COPD/Asthma', 'Chronic obstructive pulmonary disease or asthma', 'bin'],
['Hx leukemia/lymphoma', 'Leukemia or lymphoma', 'bin'],
['Hx AIDs', 'AIDS', 'bin'],
['Hx CTD', 'Connective tissue disease', 'bin'],

# diagnosis
['Sepsis/Shock', 'Septic shock', 'bin'],
['NeuroDx:IschStroke', 'Ischemic stroke', 'bin'],
['NeuroDx:HemStroke', 'Hemorrhagic stroke', 'bin'],
['NeuroDx:SAH', 'Subarachnoid hemorrhage (SAH)', 'bin'],
['NeuroDx:SDH', 'Subdural hematoma (SDH)', 'bin'],
['NeuroDx:SDH+TBI(SAH)', 'SDH or other traumatic brain injury including SAH', 'bin'],
['NeuroDx:TBI(SAH)', 'Traumatic brain injury including SAH', 'bin'],
['NeuroDx:Sz/SE', 'Seizure/status epilepticus', 'bin'],
['NeuroDx:Brain tumor', 'Brain tumor', 'bin'],
['NeuroDx:CNS infection', 'CNS infection', 'bin'],
['NeuroDx:HIE/ABI', 'Ischemic encephalopathy or Anoxic brain injury', 'bin'],
['NeuroDx:TME', 'Toxic metabolic encephalopathy', 'bin'],
['NeuroDx:Psyc', 'Primary psychiatric disorder', 'bin'],
['NeuroDx:Degenerative', 'Structural-degenerative diseases', 'bin'],
['NeuroDx:Spells', 'Spell', 'bin'],
['PrimDx:Resp', 'Respiratory disorders', 'bin'],
['PrimDx:Cardio', 'Cardiovascular disorders', 'bin'],
['PrimDx:RenalFailure', 'Kidney failure', 'bin'],
['PrimDx:LiverDisorder', 'Liver disorder', 'bin'],
['PrimDx:GI', 'Gastrointestinal disorder', 'bin'],
['PrimDx:GU', 'Genitourinary disroder', 'bin'],
['PrimDx:EndoEmegy', 'Endocrine emergency', 'bin'],
['PrimDx:NonHeadTrauma', 'Non-head trauma', 'bin'],
['PrimDx:Malignancy', 'Malignancy', 'bin'],
['PrimDx:Hem', 'Primary hematological disorder', 'bin'],
]

#['lacosamide_Hill', '', ''],
#['lacosamide_50', '', ''],
#['levetiracetam_Hill', '', ''],
#['levetiracetam_50', '', ''],
#['midazolam_Hill', '', ''],
#['midazolam_50', '', ''],
#['pentobarbital_Hill', '', ''],
#['pentobarbital_50', '', ''],
#['phenobarbital_Hill', '', ''],
#['phenobarbital_50', '', ''],
#['propofol_Hill', '', ''],
#['propofol_50', '', ''],
#['valproate_Hill', '', ''],
#['valproate_50', '', ''],

# prepare dataset
sids = pd.read_csv('potential_outcome_pkpd_c.csv')['Unnamed: 0'].values
with open('../aim1/data_to_fit_CNNIIC_iic_burden_smooth.pickle', 'rb') as ff:
    res = pickle.load(ff)
df = pd.DataFrame(data=res['C'], columns=res['Cname'], index=res['sids'])
sids_all = res['sids']
df = df.loc[sids,:].reset_index().rename(columns={'index':'SID'})
df2 = pd.read_csv('PD_Parameters_result.csv').rename(columns={'sids':'SID'})
df = df.merge(df2, on='SID', how='left')
df3 = pd.read_csv('covariates-full.csv').rename(columns={'Index':'SID'})
df = df.merge(df3[['SID', 'Race']], on='SID', how='left')

cov_names = []
values = []
for name, name_disp, data_type in cov_mappings:
    print(name, np.sum(pd.isna(df[name])))
    if data_type=='bin':
        suffix = 'n (%)'
        if name=='Race':
            if name_disp=='Asian':
                ids = df.Race=='Asian'
            elif name_disp=='Black / African American':
                ids = df.Race=='Black or African American'
            elif name_disp=='White / Caucasian':
                ids = df.Race=='White or Caucasian'
            elif name_disp=='Other':
                ids = df.Race=='Other'
            elif name_disp=='Unavailable / Declined':
                ids = pd.isna(df.Race)|np.in1d(df.Race, ['Unavailable', 'Decined'])
        else:
            assert set(df[name].dropna().unique())==set([0,1])
            ids = df[name]==1
        val = f'{np.sum(ids)} ({np.mean(ids)*100:.1f}%)'
        
    else:
        assert set(df[name].dropna().unique())!=set([0,1])
        suffix = 'median (IQR)'
        vals = df[name].astype(float).values
        val = f'{np.nanmedian(vals):.0f} ({np.nanpercentile(vals, 25):.0f} -- {np.nanpercentile(vals, 75):.0f})'
        
    cov_names.append(f'{name_disp}, {suffix}')
    values.append(val)

df_res = pd.DataFrame(data={'Variable':cov_names, 'Value':values})
df_res.to_excel('table_cov.xlsx', index=False)

