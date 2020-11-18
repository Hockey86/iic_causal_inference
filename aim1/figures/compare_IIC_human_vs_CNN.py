import os
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns


human_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output'
cnn_iic_dir = '/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt'

human_iic_sids = [x.replace('.mat','') for x in os.listdir(human_iic_dir) if x.endswith('.mat')]
cnn_iic_sids = [x.replace('.mat','') for x in os.listdir(cnn_iic_dir) if x.endswith('.mat')]

common_sids = sorted(set(human_iic_sids)&set(cnn_iic_sids), key=lambda x:int(x[len('sid'):]))
human_iics = []
cnn_iics = []
for sid in tqdm(common_sids):
    human_iic_mat = sio.loadmat(os.path.join(human_iic_dir, sid+'.mat'))
    cnn_iic_mat = sio.loadmat(os.path.join(cnn_iic_dir, sid+'.mat'))
    
    human_iic = human_iic_mat['human_iic'].flatten()
    cnn_iic = cnn_iic_mat['iic'].flatten()
    
    len_ = min(len(human_iic), len(cnn_iic))
    human_iic = human_iic[:len_]
    cnn_iic = cnn_iic[:len_]
    
    ids = (~np.isnan(human_iic))&(~np.isnan(cnn_iic))
    human_iic = human_iic[ids].astype(int)
    cnn_iic = cnn_iic[ids].astype(int)
    
    human_iics.append(human_iic)
    cnn_iics.append(cnn_iic)

human_iics = np.concatenate(human_iics)
cnn_iics = np.concatenate(cnn_iics)

cf = confusion_matrix(human_iics,cnn_iics)
cf2 = cf/cf.sum(axis=1,keepdims=True)
kappa = cohen_kappa_score(human_iics,cnn_iics)
print(kappa)

plt.close()
fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111)
sns.heatmap(np.flipud(cf2),vmin=0,vmax=1,cmap='Blues',annot=True,square=True,ax=ax,
            xticklabels=['Others','Sz','LPD','GPD','LRDA','GRDA'],
            yticklabels=['Others','Sz','LPD','GPD','LRDA','GRDA'][::-1],)
#sns.heatmap(np.flipud(cf),cmap='Blues',annot=True,square=True,ax=ax,fmt='%d', 
#            xticklabels=['Others','Sz','LPD','GPD','LRDA','GRDA'],
#            yticklabels=['Others','Sz','LPD','GPD','LRDA','GRDA'][::-1],)
ax.set_xlabel('CNN Predicted')
ax.set_ylabel('Human Annotated')
plt.tight_layout()
#plt.show()
plt.savefig('CNN_vs_human_IIC_cf.png', bbox_inches='tight', pad_inches=0.05)

