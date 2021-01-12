import sys
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress, norm, spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from myclasses import MyCalibrator, MyLogisticRegression, LTRPairwise


def bootstrap_curves(x, xs, ys, verbose=True):
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    idx = np.argsort(x)
    x_res = x[idx]
    
    ys_res = []
    for _ in tqdm(range(len(xs)), disable=not verbose):
        xx = xs[_]
        yy = ys[_]
        _, idx = np.unique(xx, return_index=True)
        idx = np.sort(idx)
        xx = xx[idx]; yy = yy[idx]
        idx = np.argsort(xx)
        xx = xx[idx]; yy = yy[idx]
        foo = interp1d(xx, yy, kind='cubic')
        ys_res.append( foo(x) )
    ys_res = np.array(ys_res)
    
    return x_res, ys_res
    

if __name__ == '__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)

    Nbt = 0
    Ncv = 5
    model_type = 'ltr'
    n_jobs = 12
    K = 7
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
    
    df_pred_all = pd.read_csv('cv_predictions_%s_Nbt%d.csv'%(model_type, Nbt))
    #df_pred_all = df_pred[df_pred.bti==0].reset_index(drop=True)
    # use all bootstraps
    ys = []; yps = []; yp_probs = []; yps_int = []
    for bti in tqdm(range(Nbt+1)):
        df_pred = df_pred_all[df_pred_all.bti==bti].reset_index(drop=True)
        ys.append( df_pred.y.values )
        yps.append( df_pred.yp.values )
        yp_probs.append( df_pred[[f'prob({x})' for x in range(K)]].values )
        yps_int.append( df_pred.yp_int.values )
    
    corrs = []
    acc0s = []; acc1s = []; acc2s = []
    for i in tqdm(range(10000)):
        ids = np.arange(len(ys[0]))
        if i>0:
            np.random.shuffle(ids)
        corrs.append( spearmanr(ys[0][ids], yps[0])[0] )
        acc0s.append( np.mean(np.abs(ys[0][ids]-yps_int[0])<=0) )
        acc1s.append( np.mean(np.abs(ys[0][ids]-yps_int[0])<=1) )
        acc2s.append( np.mean(np.abs(ys[0][ids]-yps_int[0])<=2) )
    perm_mean = np.mean(corrs[1:])
    perm_std = np.std(corrs[1:])
    pval = norm.cdf(corrs[0], perm_mean, perm_std)
    pval = min(pval,1-pval)*2
    print(f'Spearman\'s R = {corrs[0]}. Permuted = {perm_mean} (95% CI {np.percentile(corrs[1:],2.5)} -- {np.percentile(corrs[1:],97.5)}). P-value = {pval}')
    perm_mean = np.mean(acc0s[1:])
    perm_std = np.std(acc0s[1:])
    pval = norm.cdf(acc0s[0], perm_mean, perm_std)
    pval = min(pval,1-pval)*2
    print(f'acc(0) = {acc0s[0]}. Permuted = {perm_mean} (95% CI {np.percentile(acc0s[1:],2.5)} -- {np.percentile(acc0s[1:],97.5)}). P-value = {pval}')
    perm_mean = np.mean(acc1s[1:])
    perm_std = np.std(acc1s[1:])
    pval = norm.cdf(acc1s[0], perm_mean, perm_std)
    pval = min(pval,1-pval)*2
    print(f'acc(1) = {acc1s[0]}. Permuted = {perm_mean} (95% CI {np.percentile(acc1s[1:],2.5)} -- {np.percentile(acc1s[1:],97.5)}). P-value = {pval}')
    perm_mean = np.mean(acc2s[1:])
    perm_std = np.std(acc2s[1:])
    pval = norm.cdf(acc2s[0], perm_mean, perm_std)
    pval = min(pval,1-pval)*2
    print(f'acc(2) = {acc2s[0]}. Permuted = {perm_mean} (95% CI {np.percentile(acc2s[1:],2.5)} -- {np.percentile(acc2s[1:],97.5)}). P-value = {pval}')

    import pdb;pdb.set_trace()
    figsize = (8,6)

    # confusion matrix plot
    cf = confusion_matrix(ys[0], yps_int[0])
    cf2 = cf/cf.sum(axis=1,keepdims=True)*100
    plt.close()
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    sns.heatmap(cf2,annot=True,cmap='Blues')#,fmt='d')
    ax.set_ylabel('discharge mRS')
    ax.set_xlabel('Predicted discharge mRS')
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig('confusionmatrix_perc.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('confusionmatrix_perc.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
    # boxplot
    plt.close()
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111)
    ax.scatter(ys[0]+1+np.random.randn(len(ys[0]))/20, yps[0], s=20, alpha=0.2, color='k')
    ax.boxplot([yps[0][ys[0]==i] for i in range(K)], labels=range(K))
    ax.set_xlabel('discharge mRS')
    ax.set_ylabel('Predicted discharge mRS')
    sns.despine()

    plt.tight_layout()
    #plt.show()
    if display_type=='pdf':
        plt.savefig('boxplot.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('boxplot.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
    # AUC
    figsize = (8,8)
    plt.close()
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    levels = np.arange(K-1)
    #levels = [4]
    for i in levels:
        y2 = [(ys[bti]>i).astype(int) for bti in range(Nbt+1)]
        yp2 = [yp_probs[bti][:, i+1:].sum(axis=1) for bti in range(Nbt+1)]
        aucs = [roc_auc_score(y2[bti], yp2[bti]) for bti in range(Nbt+1)]
        fprs_tprs = [roc_curve(y2[bti], yp2[bti]) for bti in range(Nbt+1)]
        fprs = [x[0] for x in fprs_tprs]
        tprs = [x[1] for x in fprs_tprs]
        fpr, tprs = bootstrap_curves(fprs[0], fprs, tprs)
        auc = aucs[0]
        tpr = tprs[0]
        if Nbt>0:
            auc_lb, auc_ub = np.percentile(aucs[1:], (2.5, 97.5))
            tpr_lb, tpr_ub = np.percentile(tprs[1:], (2.5, 97.5), axis=0)
        else:
            auc_lb = np.nan
            auc_ub = np.nan
        if Nbt>0:
            ax.fill_between(fpr, tpr_lb, tpr_ub, color='k', alpha=0.4, label='95% CI')
        ax.plot(fpr, tpr, lw=2, label=f'discharge mRS <= {i} (n={np.sum(y2[0]==0)}) vs. >={i+1} (n={np.sum(y2[0]==1)}):\nAUC = {auc:.3f} [{auc_lb:.3f} - {auc_ub:.3f}]')#, c='k'
        
    ax.plot([0,1],[0,1],c='k',ls='--')
    ax.legend()
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    ax.grid(True)
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'ROCs.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'ROCs.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()

    # calibration
    plt.close()
    fig=plt.figure(figsize=figsize)
    ax=fig.add_subplot(111)
    ax.plot([0,1],[0,1],c='k',ls='--')
    levels = np.arange(K-1)
    #levels = [4]
    for i in levels:
        y2 = [(ys[bti]>i).astype(int) for bti in range(Nbt+1)]
        yp2 = [yp_probs[bti][:, i+1:].sum(axis=1) for bti in range(Nbt+1)]
        obss_preds = [calibration_curve(y2[bti], yp2[bti], n_bins=10, strategy='quantile') for bti in range(Nbt+1)]
        obss = [x[0] for x in obss_preds]
        preds = [x[1] for x in obss_preds]
        pred, obss = bootstrap_curves(preds[0], preds, obss)
        obs = obss[0]
        cslopes = [linregress(x[1], x[0])[0] for x in obss_preds]
        cslope, intercept, _, _, _ = linregress(pred, obs)
        if Nbt>0:
            cslope_lb, cslope_ub = np.percentile(cslopes[1:], (2.5, 97.5))
            obs_lb, obs_ub = np.percentile(obss[1:], (2.5, 97.5), axis=0)
        else:
            cslope_lb = np.nan
            cslope_ub = np.nan
        if Nbt>0:
            ax.fill_between(pred, obs_lb, obs_ub, color='k', alpha=0.4, label='95% CI')
        #ax.plot(pred, cslope*pred+intercept, c='k', lw=2, label=f'discharge mRS <= {i} (n={np.sum(y2[0]==0)}) vs. >={i+1} (n={np.sum(y2[0]==1)}):\ncalibration slope = {cslope:.3f} [{cslope_lb:.3f} - {cslope_ub:.3f}]')
        #ax.scatter(pred, obs, c='k', marker='o', s=40)
        ax.plot(pred, obs, lw=2, marker='o', label=f'discharge mRS <= {i} (n={np.sum(y2[0]==0)}) vs. >={i+1} (n={np.sum(y2[0]==1)}):\ncalibration slope = {cslope:.3f} [{cslope_lb:.3f} - {cslope_ub:.3f}]')#, c='k'
        
    ax.legend()
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed fraction')
    ax.grid(True)
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'calibration.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'calibration.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
