# time_bins_test.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import numpy as np, pandas as pd
from ...measures.auc import comp_auc_simple
from ...measures.point_process_measures import extract_simple_firing_rates

#compute training and testing auc for each refined time bin
def comp_split_tbin_auc(spike_time_array, df_tbins_refined, dict_tbins, **kwargs):
    """compute training and testing auc for each refined time bin.
    pcomp_split_tbin_auc returns pandas.DataFrame instance, df_tbins_refined, with
    fields auc_test and auc_train added.  it is necessary to compute both at the same time
    in order to preserve the sign associated with the corresponding roc trace.

    Example Usage:
df_tbins_refined = comp_split_tbin_auc(spike_time_array, df_tbins_refined, dict_tbins))#, **kwargs)
    """
    task_str = np.array(dict_tbins['task']['task_str'])
    cv_iter  = np.array(dict_tbins['task']['cv_iter'])
    booT_test = np.array(dict_tbins['task']['booT_test'])
    booF_test = np.array(dict_tbins['task']['booF_test'])
    booT_train = np.array(dict_tbins['task']['booT'])
    booF_train = np.array(dict_tbins['task']['booF'])

    #test that test and train are disjoint
    assert not (booT_test==booT_train)[booT_test|booT_train].any()
    assert not (booF_test==booF_train)[booF_test|booF_train].any()

    X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins_refined)
    auc_test_lst=[]
    auc_train_lst=[]
    for i,tbin_tuple in enumerate(tbin_indices):
        #compute training auc for candidate time bin
        predictor_values = np.concatenate((X[booT_train,i],X[booF_train,i]))
        label_values = np.zeros(predictor_values.shape[0],dtype=int)
        label_values[:sum(booT_train)]=1
        auc_train=comp_auc_simple(label_values, predictor_values)
        #compute testing auc for candidate time bin
        predictor_values = np.concatenate((X[booT_test,i],X[booF_test,i]))
        label_values = np.zeros(predictor_values.shape[0],dtype=int)
        label_values[:sum(booT_test)]=1
        auc_test=comp_auc_simple(label_values, predictor_values)
        #rectify sign consistent with training
        if auc_train<0.5:
            auc_train=1-auc_train
            auc_test=1-auc_test
        auc_test_lst.append(auc_test)
        auc_train_lst.append(auc_train)
    df_tbins_refined['auc_test']=auc_test_lst
    df_tbins_refined['auc_train']=auc_train_lst
    return df_tbins_refined
