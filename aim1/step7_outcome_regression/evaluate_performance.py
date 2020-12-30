import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
from myclasses import MyCalibrator, MyLogisticRegression, LTRPairwise

#data_type = 'humanIIC'
data_type = 'CNNIIC'

response_tostudy = 'iic_burden_smooth'
#response_tostudy = 'spike_rate'

Nbt = 1#000
Ncv = 5
model_type = 'ltr'
n_jobs = 12
random_state = 2020

with open('results_%s_Nbt%d.pickle'%(model_type, Nbt), 'rb') as ff:
    res = pickle.load(ff)
for k in res:
    exec(f'{k} = res["{k}"]')
#print(params)

if Nbt>0:
    for idx in tr_scores_bt[0].index:
        print(idx)
        print('te score: %f [%f -- %f]'%(
            te_scores_bt[0][idx],
            np.percentile([x[idx] for x in te_scores_bt[1:]], 2.5),
            np.percentile([x[idx] for x in te_scores_bt[1:]], 97.5),))
else:
    print('te score:', te_scores_bt[0])

df_coef = pd.DataFrame(data={'Xname':Xnames, 'coef':coefs_bt[0]})
df_coef = df_coef.sort_values('coef', ascending=False).reset_index(drop=True)
df_coef.to_csv('coef.csv', index=False)

df_y_yp = pd.read_csv('cv_predictions_%s_Nbt%d.csv'%(model_type, Nbt))
df_y_yp = df_y_yp[df_y_yp.bti==0].reset_index(drop=True)

K = 7
figsize = (8,6)
plt.close()
fig = plt.figure(figsize=figsize)

ax = fig.add_subplot(111)
ax.boxplot([df_y_yp.yp[df_y_yp.y==i] for i in range(K)], labels=range(K))
ax.set_xlabel('Actual discharge mRS')
ax.set_ylabel('Predicted discharge mRS')
seaborn.despine()

plt.tight_layout()
#plt.show()
plt.savefig('boxplot.png', bbox_inches='tight', pad_inches=0.05)
