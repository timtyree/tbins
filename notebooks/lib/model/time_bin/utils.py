# time_bin.utils.py
#Programmer: Tim Tyree
#Date: 9.8.2022
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
from .test import comp_split_tbin_auc
from .time_bin_main import gener_tbins_fast
from ...measures.auc import comp_auc_simple
from ...measures.point_process_measures import extract_simple_firing_rates

def gener_tbin_filter(df_tbins,
                      max_frac_trials_anomalously_low_fr=0.2,
                      max_dVpp=15, #mV
                      max_final_err_1_streak=1,
                      max_training_p_aggregated=0.01,
                      frac_drop_training_auc=0.5,
                      min_training_auc=None,**kwargs):
    """gener_tbin_filter returns a boolean index of pandas.DataFrame instance, df_tbins.
    boo_dVpp identifies underplit units instead of using error code 3 abundance.
    boo_low_fr,boo_fstreak identifies overplit units using error code 1 abundance.
    frac_drop_training_auc is only used if min_training_auc is not None.

    Example Usage:
boo_low_fr,boo_dVpp,boo_fstreak,boo_pagg,boo_low_training_auc = gener_tbin_filter(
                      df_tbins = df_pred_tbins_agg,
                      max_frac_trials_anomalously_low_fr=0.2,max_dVpp=15, #mV
                      max_final_err_1_streak=1,max_training_p_aggregated=0.01,
                      frac_drop_training_auc=0.5,min_training_auc=None)
boo_filter =  boo_low_fr|boo_dVpp|boo_fstreak|boo_pagg|boo_low_training_auc
sum(boo_fstreak),sum(boo_low_fr&~boo_dVpp),sum(~boo_low_fr&boo_dVpp),sum(boo_low_training_auc),df_pred_tbins_agg.shape[0]
    """
    if min_training_auc is None:
        min_training_auc=np.quantile(df_tbins['auc_train'],frac_drop_training_auc)

    boo_low_fr=df_tbins['frac_trials_anomalously_low_fr']>=max_frac_trials_anomalously_low_fr
    boo_dVpp=df_tbins['dVpp']>=max_dVpp
    boo_fstreak = df_tbins['final_err_1_streak']>=max_final_err_1_streak
    boo_pagg = df_tbins['p_aggregated']>=max_training_p_aggregated
    boo_low_training_auc= df_tbins['auc_train']<=min_training_auc
    return boo_low_fr,boo_dVpp,boo_fstreak,boo_pagg,boo_low_training_auc

def extract_pred_tbin_list_detailed(dict_tbins_set, df_n, df_es, count_err_3_by_neuron, spike_time_array,
    p_significant=0.05,printing=True,
    use_reasonable_filtering=False,
    max_frac_trials_anomalously_low_fr=0.2,max_dVpp=15, #mV
    max_final_err_1_streak=1,max_training_p_aggregated=0.01,
    frac_drop_training_auc=0.5,min_training_auc=None,**kwargs):
    """extract_pred_tbin_list_detailed extracts predictive tbins from dict_tbins_set
    and populates each time bin with a supplementary of training information.
    kwargs following use_reasonable_filtering are not used if use_reasonable_filtering is False.
    if use_reasonable_filtering is true, then extract_pred_tbin_list_detailed will attempt to
    identify predictive time bins that are unreasonable (e.g. low firing rate, amplitude too high, training auc low, etc...).

    Example Usage:
df_pred_tbins_lst = extract_pred_tbin_list_detailed(dict_tbins_set, df_n, df_es, count_err_3_by_neuron, spike_time_array,
            p_significant=0.05,printing=True,
            use_reasonable_filtering=False,
            max_frac_trials_anomalously_low_fr=0.2,max_dVpp=15, #mV
            max_final_err_1_streak=1,max_training_p_aggregated=0.01,
            frac_drop_training_auc=0.5,min_training_auc=None)#,**kwargs)
    """
    if printing:
        print(f"extracting descriptive properties of predictive time bins...")
        print(f"overall AUC: \t\t\tmin,\t25%,\tmedian\t75%,\tmax")
    df_pred_tbins_lst=[]
    for dict_tbins in dict_tbins_set['dict_tbins_lst']:
        df_tbins_refined = pd.DataFrame(dict_tbins['df_tbins_refined'])
        cv_iter=dict_tbins['task']['cv_iter']
        task_str=dict_tbins['task']['task_str']
        preamble=f"{task_str} (cv_iter: {cv_iter}):\t"

        #identify a fraction of trials with anomolously low fr for each predictive neuron
        nid_values=df_tbins_refined['nid'].values
        df_tbins_refined.set_index('nid',drop=True,inplace=True)
        df_tbins_refined.loc[:,'frac_trials_anomalously_low_fr']  = df_n['frac_trials_anomalously_low_fr']
        df_tbins_refined.loc[:,'GlobalIndex']=df_n['GlobalIndex']
        #give dVpp to df_tbins_refined
        df_tbins_refined.loc[:,'dVpp']=df_n['dVpp']
        #give num_trials_max_fr_exceeded to df_tbins_refined
        err_3_count = count_err_3_by_neuron[nid_values]
        df_tbins_refined['err_3_count']=err_3_count
        #give warnings streaks of not firing to df_tbins_refined
        df_tbins_refined['longest_err_1_streak']=df_es['longest_error_streak']
        df_tbins_refined['final_err_1_streak']=df_es['final_error_streak']
        df_tbins_refined['max_spike_count']=df_es['max_spike_count']
        df_tbins_refined.reset_index(inplace=True)
        #compute the training/testing auc as label
        df_tbins_refined = comp_split_tbin_auc(spike_time_array, df_tbins_refined, dict_tbins)
        df_tbins_refined['cv_iter']=cv_iter
        df_tbins_refined['task_str']=task_str
        #select predictive time bins
        df_pred_tbins= df_tbins_refined[df_tbins_refined['p']<p_significant].copy()
        if printing:
            df_descr = df_pred_tbins[['auc_overall']].describe().T[['min','25%','50%','75%','max']]
            mean_values = df_descr.mean().values
            print(preamble+f"  num pred neurons: {df_pred_tbins.shape[0]}, {mean_values}")

        if use_reasonable_filtering:
            boo_low_fr,boo_dVpp,boo_fstreak,boo_pagg,boo_low_training_auc = gener_tbin_filter(
                          df_tbins = df_pred_tbins,
                          max_frac_trials_anomalously_low_fr=max_frac_trials_anomalously_low_fr,max_dVpp=max_dVpp,
                          max_final_err_1_streak=max_final_err_1_streak,max_training_p_aggregated=max_training_p_aggregated,
                          frac_drop_training_auc=frac_drop_training_auc,min_training_auc=min_training_auc)
            boo_filter =  boo_low_fr|boo_dVpp|boo_fstreak|boo_pagg|boo_low_training_auc
            df_pred_tbins = df_pred_tbins[~boo_filter].copy()
            if printing:
                df_descr = df_pred_tbins[['auc_overall']].describe().T[['min','25%','50%','75%','max']]
                mean_values = df_descr.mean().values
                print(preamble+f"  num pred neurons: {df_pred_tbins.shape[0]}, {mean_values} (filtered)")

        #record
        df_pred_tbins_lst.append(df_pred_tbins.copy())
    return df_pred_tbins_lst
