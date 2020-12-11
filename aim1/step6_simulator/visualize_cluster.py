import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.impute import KNNImputer
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


# load cluster id
clusterid = pd.read_csv('Cluster.csv', header=None).values
clusterid = np.argmax(clusterid, axis=1)  # ranges from 0 to Ncluster-1

# load covariates
mat = sio.loadmat('C.mat')
C = mat['C']
Cname = [x.strip() for x in mat['Cname'].flatten()]
remove_id = Cname.index('APACHE II  first 24')
C = C[:,np.arange(C.shape[1])!=remove_id]
Cname = [Cname[i] for i in range(len(Cname)) if i!=remove_id]
print(Cname)

# impute missing
C = KNNImputer(n_neighbors=5).fit_transform(C)
assert ~np.any(np.isnan(C))

# plot umap
#reducer = umap.UMAP()
reducer = TSNE()
Cumap = reducer.fit_transform(C)
colors = 'rgbkycm'
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
for ci in range(7):
    ax.scatter(Cumap[clusterid==ci][:,0], Cumap[clusterid==ci][:,1], color=colors[ci], s=20)
plt.tight_layout()
#plt.show()
plt.savefig('figures/cluster_vis/cluster_vis_tsne.png')
import pdb;pdb.set_trace()

# plot boxplot
for feat_id in tqdm(range(len(Cname))):

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    groups = [C[clusterid==i][:,feat_id] for i in range(7)]
    ax.boxplot(groups)
    ax.set_ylabel(Cname[feat_id])

    plt.tight_layout()
    #plt.show()
    feat_name = Cname[feat_id].replace('/', '_')
    plt.savefig('figures/cluster_vis/cluster_vis_%s.png'%feat_name)

