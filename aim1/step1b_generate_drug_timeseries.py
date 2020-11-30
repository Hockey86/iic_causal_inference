import datetime
import glob
import os
import re
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
import pandas as pd
from tqdm import tqdm


window_time = 10  # [s]
step_time = 2   # [s]
pad_time = 4    # [s]


if __name__=='__main__':
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
    #output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/generate_drug_data_to_crosscheck_with_Rajesh'
    output_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data'
   
    master_list = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data/SAGE_DataScrub_SBullock_11.4.2019_HaoqiCorrected.csv')
    master_list['The start day of first EEG'] = pd.to_datetime(master_list['The start day of first EEG'])
    
    # drug table
    
    drug_df = pd.read_csv(os.path.join(output_dir, 'drug_data_2000pts.csv'))
    drug_df.loc[drug_df.drugNameSimplified=='phenytoin', 'drugNameSimplified'] = 'fosphenytoin'
    drug_df.loc[drug_df.drugNameSimplified=='divalproex', 'drugNameSimplified'] = 'valproate'
    drug_df.Admin_Time = pd.to_datetime(drug_df.Admin_Time, format='%Y-%m-%dT%H:%M:%S')
    drug_df.Dose_Unit = drug_df.Dose_Unit.str.upper()
    drug_df.loc[pd.isna(drug_df.Dose_Unit), 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG_PE', 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG PE', 'Dose_Unit'] = 'MG'
    drug_df.loc[drug_df.Dose_Unit=='MG PE/KG', 'Dose_Unit'] = 'MG/KG'
    print(set(drug_df.Dose_Unit))
    
    # get label
    #human_label_dir = '/home/sunhaoqi/Desktop/IIC_human_labels'
    #human_label_paths = glob.glob(os.path.join(human_label_dir, '*.csv'))
    label_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/data_score/cnn_label_24h_2000pt'
    label_paths = os.listdir(label_dir)
    
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
    sids = []
    notfound_sids = []
    mrns = []
    group1_paths = os.listdir('/media/mad3/Projects/SAGE_Data/Group1')
    group2_paths = os.listdir('/media/mad3/Projects/SAGE_Data/Group2')
    for i in tqdm(range(len(master_list))):
        sid2 = 'sid%04d'%int(master_list.Index.iloc[i][3:])
        #sid_path = os.path.join('/media/mad3/Projects/SAGE_Data', sid2, 'Data')
        #eeg_file_names = os.listdir(sid_path)
        sid_path = None
        for group_name in ['Group1', 'Group2']:
            group_paths = eval(group_name.lower()+'_paths')
            group_ids = np.where([re.search(sid2+'[a-z]*', x, re.IGNORECASE) is not None for x in group_paths])[0]
            if len(group_ids)==1:
                sid_path = os.path.join('/media/mad3/Projects/SAGE_Data', group_name, group_paths[group_ids[0]], 'Data')
                break
            for group_id in group_ids:
                this_sid_path = os.path.join('/media/mad3/Projects/SAGE_Data', group_name, group_paths[group_id], 'Data')
                start_times = [datetime.datetime.strptime(x[x.find('_'):], '_%Y%m%d_%H%M%S.mat') for x in os.listdir(this_sid_path)]
                if master_list['The start day of first EEG'].iloc[i].date()==min(start_times).date():
                    sid_path = this_sid_path
                    break
            if sid_path is not None:
                break
        if sid_path is None:
            notfound_sids.append(master_list.Index.iloc[i])
            continue
        eeg_file_names = os.listdir(sid_path)
        
        eeg_start_time = min([datetime.datetime.strptime(x[x.find('_'):], '_%Y%m%d_%H%M%S.mat') for x in eeg_file_names])
        
        ids = np.where(df_bodyweights.MRN==master_list.MRN.iloc[i])[0]
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
        ages.append(master_list.Age.iloc[i])
        gender = master_list.Gender.iloc[i]
        genders.append(1 if gender=='M' else 0)
        sids.append(master_list.Index.iloc[i])
        mrns.append(master_list.MRN.iloc[i])
        eeg_start_times.append(eeg_start_time)
    bodyweights = np.array(bodyweights)
    bodyweights[bodyweights<=20] = np.nan
    ages = np.array(ages)
    genders = np.array(genders)
    print(f'{len(notfound_sids)} subjects are not found: {notfound_sids}')
    
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
    
    sids_no_drug = []
    sids_no_iic = []
    output_dir = os.path.join(output_dir, 'drug_timeseries_2000pts')
    for si, sid in enumerate(tqdm(sids)):
        output_path = os.path.join(output_dir, '%s_2secWindow.mat'%sid)
        #if os.path.exists(output_path):
        #    continue
        drug_df_ = drug_df[drug_df.MRN==mrns[si]]
        if len(drug_df_)==0:
            sids_no_drug.append(sid)
            continue
        drug_df_ = drug_df_.sort_values('Admin_Time').reset_index(drop=True)
        eeg_start_time = eeg_start_times[si]
        
        #labels = pd.read_csv([x for x in human_label_paths if 'sid%04d'%int(sid[3:]) in x][0], header=None)
        this_label_paths = [x for x in label_paths if x.startswith(sid+'_')]
        if len(this_label_paths)==0:
            sids_no_iic.append(sid)
            continue
        label_start_times = [datetime.datetime.strptime(x[x.find('_'):],'_%Y%m%d_%H%M%S.npy')+datetime.timedelta(seconds=pad_time) for x in this_label_paths]
        label_start_time = min(label_start_times)
        label_end_time = max(label_start_times)
        dur = np.load(os.path.join(label_dir, this_label_paths[np.argmax(label_start_times)])).shape[0]*2
        label_end_time += datetime.timedelta(seconds=dur)
        T = int(np.floor((label_end_time-label_start_time).total_seconds()/step_time))
            
        # generate a time series based on drug_df_ (no data is 0)
        #drug_ts = {}
        drugnames = set(drug_df_.drugNameSimplified)
        res = {}
        for drugname in drugnames:
            this_drug_df_ = drug_df_[drug_df_.drugNameSimplified==drugname].reset_index(drop=True)
            this_drug_ts = np.zeros(T)
            for i in range(len(this_drug_df_)):
                if this_drug_df_.Dose_Unit.iloc[i] in ['MG', 'MG/KG', 'MCG']:
                    start = int(round((this_drug_df_.Admin_Time.iloc[i]-eeg_start_time).total_seconds()/step_time))
                    end = start+60//step_time
                    if this_drug_df_.Dose_Unit.iloc[i] == 'MCG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60/1000.
                    elif this_drug_df_.Dose_Unit.iloc[i] == 'MG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60
                    elif this_drug_df_.Dose_Unit.iloc[i] == 'MG/KG':
                        dose = this_drug_df_.Dose_Amount.iloc[i]*60*bodyweights[si]
                        
                elif this_drug_df_.Dose_Unit.iloc[i] in ['MG/HR', 'MG/KG/HR', 'MCG/KG/HR', 'MCG/KG/MIN']:
                    start = int(round((this_drug_df_.Admin_Time.iloc[i]-eeg_start_time).total_seconds()/step_time))
                    if i==len(this_drug_df_)-1:
                        end = len(this_drug_ts)
                    else:
                        end = int(round((this_drug_df_.Admin_Time.iloc[i+1]-eeg_start_time).total_seconds()/step_time))
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
            
            res[drugname+'_dose'] = csr_matrix(this_drug_ts)
            res[drugname+'_dose_bodyweight_normalized'] = csr_matrix(this_drug_ts/bodyweights[si])
            
        # fill nans to gaps
        #this_drug_ts[nan_ids] = np.nan
        
        # save into mat
        res.update({'label_start_time': datetime.datetime.strftime(label_start_time, '%Y/%m/%d %H:%M:%S'),
                   'eeg_start_time': datetime.datetime.strftime(eeg_start_time, '%Y/%m/%d %H:%M:%S')})
        sio.savemat(output_path, res)
        
    print(f'{len(sids_no_drug)} subjects does not have matching drug records: {sids_no_drug}')
    print(f'{len(sids_no_iic)} subjects does not have matching IIC labels: {sids_no_iic}')
            
    """
    458 subjects does not have matching drug records: ['sid1', 'sid10', 'sid11', 'sid19', 'sid20', 'sid23', 'sid25', 'sid29', 'sid34', 'sid37', 'sid38', 'sid43', 'sid44', 'sid45', 'sid46', 'sid47', 'sid49', 'sid53', 'sid58', 'sid62', 'sid64', 'sid65', 'sid68', 'sid72', 'sid75', 'sid79', 'sid84', 'sid85', 'sid97', 'sid101', 'sid103', 'sid105', 'sid106', 'sid108', 'sid112', 'sid114', 'sid115', 'sid118', 'sid119', 'sid129', 'sid134', 'sid135', 'sid137', 'sid141', 'sid147', 'sid157', 'sid158', 'sid159', 'sid161', 'sid170', 'sid171', 'sid173', 'sid175', 'sid176', 'sid179', 'sid188', 'sid191', 'sid194', 'sid200', 'sid202', 'sid206', 'sid211', 'sid213', 'sid217', 'sid218', 'sid220', 'sid221', 'sid222', 'sid223', 'sid229', 'sid230', 'sid233', 'sid235', 'sid243', 'sid244', 'sid246', 'sid247', 'sid248', 'sid249', 'sid255', 'sid256', 'sid257', 'sid259', 'sid275', 'sid276', 'sid279', 'sid281', 'sid284', 'sid309', 'sid312', 'sid313', 'sid319', 'sid321', 'sid324', 'sid333', 'sid343', 'sid346', 'sid349', 'sid354', 'sid356', 'sid361', 'sid368', 'sid370', 'sid371', 'sid372', 'sid380', 'sid382', 'sid389', 'sid397', 'sid399', 'sid401', 'sid405', 'sid407', 'sid408', 'sid414', 'sid419', 'sid448', 'sid457', 'sid501', 'sid548', 'sid587', 'sid588', 'sid595', 'sid597', 'sid600', 'sid610', 'sid614', 'sid618', 'sid625', 'sid626', 'sid629', 'sid633', 'sid634', 'sid640', 'sid645', 'sid647', 'sid649', 'sid651', 'sid654', 'sid659', 'sid660', 'sid664', 'sid670', 'sid672', 'sid680', 'sid688', 'sid690', 'sid695', 'sid702', 'sid703', 'sid709', 'sid711', 'sid712', 'sid733', 'sid735', 'sid739', 'sid743', 'sid744', 'sid746', 'sid750', 'sid754', 'sid755', 'sid757', 'sid759', 'sid761', 'sid763', 'sid765', 'sid770', 'sid771', 'sid778', 'sid779', 'sid781', 'sid783', 'sid787', 'sid788', 'sid791', 'sid793', 'sid794', 'sid795', 'sid796', 'sid797', 'sid803', 'sid806', 'sid807', 'sid808', 'sid811', 'sid813', 'sid822', 'sid828', 'sid829', 'sid830', 'sid859', 'sid863', 'sid864', 'sid865', 'sid875', 'sid889', 'sid890', 'sid891', 'sid892', 'sid907', 'sid908', 'sid913', 'sid927', 'sid961', 'sid963', 'sid968', 'sid973', 'sid975', 'sid976', 'sid977', 'sid978', 'sid982', 'sid984', 'sid985', 'sid986', 'sid989', 'sid991', 'sid993', 'sid997', 'sid998', 'sid1000', 'sid1003', 'sid1004', 'sid1005', 'sid1008', 'sid1009', 'sid1010', 'sid1011', 'sid1015', 'sid1045', 'sid1080', 'sid1084', 'sid1093', 'sid1101', 'sid1102', 'sid1105', 'sid1137', 'sid1140', 'sid1145', 'sid1150', 'sid1152', 'sid1159', 'sid1164', 'sid1170', 'sid1175', 'sid1176', 'sid1177', 'sid1178', 'sid1179', 'sid1181', 'sid1182', 'sid1186', 'sid1187', 'sid1195', 'sid1196', 'sid1202', 'sid1205', 'sid1206', 'sid1214', 'sid1215', 'sid1226', 'sid1230', 'sid1232', 'sid1239', 'sid1246', 'sid1249', 'sid1250', 'sid1251', 'sid1262', 'sid1263', 'sid1271', 'sid1273', 'sid1277', 'sid1278', 'sid1286', 'sid1288', 'sid1291', 'sid1299', 'sid1302', 'sid1303', 'sid1310', 'sid1319', 'sid1322', 'sid1326', 'sid1339', 'sid1340', 'sid1347', 'sid1351', 'sid1354', 'sid1356', 'sid1357', 'sid1366', 'sid1367', 'sid1371', 'sid1379', 'sid1388', 'sid1389', 'sid1391', 'sid1394', 'sid1396', 'sid1403', 'sid1405', 'sid1407', 'sid1408', 'sid1410', 'sid1411', 'sid1416', 'sid1417', 'sid1423', 'sid1424', 'sid1432', 'sid1433', 'sid1448', 'sid1449', 'sid1451', 'sid1454', 'sid1459', 'sid1462', 'sid1463', 'sid1465', 'sid1467', 'sid1469', 'sid1470', 'sid1485', 'sid1487', 'sid1488', 'sid1496', 'sid1504', 'sid1505', 'sid1508', 'sid1511', 'sid1515', 'sid1521', 'sid1524', 'sid1529', 'sid1533', 'sid1536', 'sid1538', 'sid1539', 'sid1541', 'sid1542', 'sid1550', 'sid1554', 'sid1557', 'sid1560', 'sid1566', 'sid1567', 'sid1569', 'sid1571', 'sid1572', 'sid1579', 'sid1580', 'sid1581', 'sid1582', 'sid1585', 'sid1586', 'sid1587', 'sid1588', 'sid1590', 'sid1592', 'sid1596', 'sid1598', 'sid1604', 'sid1609', 'sid1611', 'sid1612', 'sid1613', 'sid1622', 'sid1626', 'sid1629', 'sid1640', 'sid1643', 'sid1645', 'sid1649', 'sid1650', 'sid1653', 'sid1666', 'sid1673', 'sid1690', 'sid1693', 'sid1694', 'sid1702', 'sid1705', 'sid1706', 'sid1707', 'sid1717', 'sid1720', 'sid1722', 'sid1727', 'sid1728', 'sid1730', 'sid1735', 'sid1736', 'sid1740', 'sid1744', 'sid1745', 'sid1746', 'sid1747', 'sid1748', 'sid1749', 'sid1753', 'sid1762', 'sid1767', 'sid1769', 'sid1774', 'sid1776', 'sid1778', 'sid1785', 'sid1788', 'sid1793', 'sid1801', 'sid1802', 'sid1806', 'sid1813', 'sid1823', 'sid1824', 'sid1830', 'sid1833', 'sid1835', 'sid1843', 'sid1845', 'sid1847', 'sid1852', 'sid1853', 'sid1855', 'sid1860', 'sid1862', 'sid1866', 'sid1874', 'sid1891', 'sid1896', 'sid1900', 'sid1902', 'sid1909', 'sid1910', 'sid1911', 'sid1912', 'sid1931', 'sid1936', 'sid1937', 'sid1938', 'sid1952', 'sid1962', 'sid1973', 'sid1974', 'sid1977', 'sid1979', 'sid1982', 'sid1987', 'sid1988', 'sid1992', 'sid1993', 'sid1994', 'sid1995', 'sid1996', 'sid1998', 'sid2000']
    0 subjects does not have matching IIC labels: []
    """
