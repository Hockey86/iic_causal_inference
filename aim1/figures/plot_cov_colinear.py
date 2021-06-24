import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 9})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
seaborn.set_style('ticks')


def cluster_corr(corr_array):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    return corr_array[idx, :][:, idx], idx


def binary_correlation(x, y):
    """
    Correlation for binary vectors (0 or 1)
    Ref:
    [1] Zhang, B. and Srihari, S.N., 2003, September. Properties of binary vector dissimilarity measures. In Proc. JCIS Int'l Conf. Computer Vision, Pattern Recognition, and Image Processing (Vol. 1). https://cedar.buffalo.edu/papers/articles/CVPRIP03_propbina.pdf
    [2] Tubbs, J.D., 1989. A note on binary template matching. Pattern Recognition, 22(4), pp.359-365.
    """
    #corr = np.mean(X[:,i]==X[:,j])*2-1
    #corrpearsonr(X[:,i], X[:,j])[0]
    
    s11 = np.sum(x*y)
    s10 = np.sum(x*(1-y))
    s01 = np.sum((1-x)*y)
    s00 = np.sum((1-x)*(1-y))
    
    # The "Correlation" method in Table 1 of [1].
    sigma = np.sqrt( (s10+s11)*(s01+s00)*(s11+s01)*(s00+s10) )
    corr = (s11*s00-s10*s01)/sigma
    
    """
    # The "Rogers-Tanmot" method in Table 1 of [1].
    corr = (s11+s00)/(s11+s00+2*s10+2*s01)
    corr = corr*2-1  # turn it into -1 to 1
    """
    
    return corr
    
    
if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
    
    df = pd.read_csv('/data/Dropbox (Partners HealthCare)/CausalModeling_IIIC/data_to_share/step1_output_2000pt/covariates-to-be-used.csv')
    Xnames = list(df.columns)
    Xnames.remove('Index')
    Xnames = np.array(Xnames)
    
    X = df[Xnames].values.astype(float)
    corr = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            ids = (~np.isnan(X[:,i]))&(~np.isnan(X[:,j]))
            corr[i,j] = spearmanr(X[:,i][ids], X[:,j][ids])[0]
    corr, idx = cluster_corr(corr)
    
    figsize = (25, 16.5)
    panel_xoffset = -0.1
    panel_yoffset = 1
    
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    corr_linkage = sch.linkage(corr+1, 'ward')
    dendro = sch.dendrogram( corr_linkage, ax=ax1, labels=Xnames[idx])#, truncate_mode='level', p=10)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax1.set_ylabel('Distance')
    ax1.set_yticks([])
    ax1.set_xticks(dendro_idx*10)
    ax1.set_xticklabels(dendro['ivl'], rotation=-90, fontsize=10)#, ha='left'
    seaborn.despine()
    ax1.text(-0.04, panel_yoffset, 'A', ha='right', va='top', transform=ax1.transAxes, fontweight='bold')
    
    im = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']], cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation=-90)#, ha='left')
    ax2.set_yticklabels(dendro['ivl'])
    """
    ids = [list(Xnames).index(x) for x in [
        'GPD/BIPD', 'G delta slowing', 'G theta slowing',  'GRDA',  'G NCSE', 'G Sz',
        'Asymmetry', 'LPD', 'LRDA', 'F NCSE', 'BIRDs',
        'IBA', 'MLV', 'ELV','BS w spike', 'BS w/o spike',
        'EDB', 'Unreactive',
        'No sleep pattern',]]
    im = ax2.imshow(corr[ids][:,ids], cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(corr.shape[1]))
    ax2.set_yticks(np.arange(corr.shape[0]))
    ax2.set_xticklabels(Xnames[ids], rotation=-55, ha='left')
    ax2.set_yticklabels(Xnames[ids])
    """
    ax2.text(-0.3, panel_yoffset, 'B', ha='right', va='top', transform=ax2.transAxes, fontweight='bold')
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax)#, orientation='horizontal')
    cbar.ax.set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    if display_type=='pdf':
        plt.savefig('corr_mat.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('corr_mat.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
