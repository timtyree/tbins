# distinguishability.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu,kstest

def estimate_distinguishability_tbin_crossval(X,y,tbin_index,n_splits=5,random_state=42,shuffle=True,ptest=None,use_mannwhitneyu=True,printing=False,**kwargs):
    """estimate_distinguishability_tbin_crossval returns a pandas.DataFrame instance for each fold for each tbin.
    estimate_distinguishability_tbin_crossval generates stratified-cross-validated p-values for general features.
    and returns them as a pandas.DataFrame instance with field 'tbin_index' describing the column/feature index of X.
    ptest is the statistical test used to estimate the distinguishability of each fold.
    if ptest is not None, then scipy.stats.mannwhitneyu is used if use_mannwhitneyu is True; otherwise, kstest is used.
    X: 2D numpy.array instance with rows indexing trials and columns indexing features.
    y: 1D numpy.array instance with 0 for false and y=1 for true.  y=nan for trials to be ignored.
    n_splits≥2: the integer number of folds considered by stratified cross-validation over the training trials.

    Example Usage:
y=np.empty(X.shape[0])*np.nan
y[booT]=1.
y[booF]=0.
df_cv=estimate_distinguishability_tbin_crossval(X,y,tbin_index,n_splits=5)#,random_state=42,shuffle=True,ptest=None,use_mannwhitneyu=True,printing=False,**kwargs)
    """
    assert n_splits>=2
    i=tbin_index
    if ptest is None:
        if use_mannwhitneyu:
            ptest=mannwhitneyu
        else:
            ptest=kstest
    else:
        ptest=False
    booT=np.isclose(y,1.)
    booF=np.isclose(y,0.)
    if printing:
        print(f"num. match trials: {sum(booT)}, num. mismatch trials: {sum(booF)}")

    #test the aggregated distinguishability
    fr_values_match=X[booT,i]
    fr_values_mismatch=X[booF,i]
    _,p_kstest_aggregated=ptest(fr_values_match,fr_values_mismatch)

    boo=booT|booF
    FR_values=X[boo,i].flatten()#.reshape(-1)
    y_values=y[boo]
    assert y_values.shape[0]>=n_splits

    #compute p value for each fold
    dict_cv_lst=[]
    kf = StratifiedKFold (shuffle=shuffle,n_splits=n_splits,random_state=random_state)
    for train_index, test_index in kf.split(FR_values,y_values):
        y_=y_values[train_index]
        X_=FR_values[train_index]
        booM_=np.isclose(y_,1.)
        booMM_=np.isclose(y_,0.)
        fr_values_match=X_[booM_]
        fr_values_mismatch=X_[booMM_]
        if (fr_values_match.shape[0]>0) and (fr_values_mismatch.shape[0]>0):
            _,p_kstest=ptest(fr_values_match,fr_values_mismatch)
            num_trialsF=sum(booMM_)
            num_trialsT=sum(booM_)
            dict_cv=dict(
                tbin_index=i,
                p=p_kstest,
                pagg=p_kstest_aggregated,
                num_trialsT=num_trialsT,
                num_trialsF=num_trialsF,
                use_mannwhitneyu=use_mannwhitneyu,
            )
            dict_cv_lst.append(dict_cv)
    assert len(dict_cv_lst)>0
    df_cv=pd.DataFrame(dict_cv_lst)
    if printing:
        maxp_kstest=df_cv['p'].max()
        print(f"num. match trials: {num_trialsT}, num. mismatch trials: {num_trialsF}, max p-value: {maxp_kstest:.4f}")
    return df_cv

def estimate_distinguishability_tbins(X,tbin_indices,booT,booF,n_splits=5,mode='mean',random_state=42,shuffle=True,ptest=None,use_mannwhitneyu=True,printing=False,**kwargs):
    """estimate_distinguishability_tbins performs K-fold cross-validation on predictive time bins, where K=n_splits,
    and returns results from cross validation as a tuple.
    mode indicates the method used to summarize the auc values of the training folds for each time bin.
    mode has options: {'mean','median','max','min'}.
    kwargs are passed to estimate_distinguishability_tbin_crossval directly.
    estimate_distinguishability_tbins performs the following:
        1. perform K-fold cross validation with each time bin
        2. make dictionary from tbin_index to df_cv as dict_tbins_cv_lst
        3. summarize tbin pvalues as df_tbins_cv using method indicated by mode.

    Example Usage:
X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins)
df_tbins_cv,dict_tbins_cv_lst=estimate_distinguishability_tbinsX,tbin_indices,booT,booF,n_splits=5)
    """
    num_trials = X.shape[0]
    num_tbins = X.shape[1]
    y=np.empty(num_trials)*np.nan
    y[booT]=1.
    y[booF]=0.
    df_cv_lst=[]
    for i in range(num_tbins):
        #estimate p-value distinguishability for each crossvalidation fold
        df_cv=estimate_distinguishability_tbin_crossval(X,y,tbin_index=i,n_splits=n_splits,
                                    random_state=random_state,shuffle=shuffle,ptest=ptest,
                                    use_mannwhitneyu=use_mannwhitneyu,printing=printing,**kwargs)
        #record
        df_cv_lst.append(df_cv)
    if printing:
        print(f"distinguishability estimation complete!")
    #summarizing df_cv as p
    dict_tbins_cv_lst=[]
    for i,df_cv in enumerate(df_cv_lst):
        nid,tau1,tau2,lop=tbin_indices[i]
        #compute the mean number of trials
        num_trialsT=df_cv['num_trialsT'].mean()
        num_trialsF=df_cv['num_trialsF'].mean()
        #compute the representative p-value
        if mode=='mean':
            #compute the mean p-value
            p=df_cv['p'].mean()
            p_aggregated=df_cv['pagg'].mean()
        elif mode=='median':
            p=df_cv['p'].median()
            p_aggregated=df_cv['pagg'].median()
        elif mode=='max':
            p=df_cv['p'].max()
            p_aggregated=df_cv['pagg'].max()
        elif mode=='min':
            p=df_cv['p'].min()
            p_aggregated=df_cv['pagg'].min()
        else:
            raise("Input Error: mode not yet implemented in cross_validate_predictive_tbins!")
        #format data
        dict_tbins_cv=dict(
            tbin_index=i,
            p=p, #representative p-value
            p_aggregated=p_aggregated,
            num_trialsT=num_trialsT,
            num_trialsF=num_trialsF,
            nid=int(nid),
            tau1=tau1,
            tau2=tau2
        )
        #record
        dict_tbins_cv_lst.append(dict_tbins_cv)
    #format
    df_tbins_cv=pd.DataFrame(dict_tbins_cv_lst)
    return df_tbins_cv,dict_tbins_cv_lst
