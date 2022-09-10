#time_bin.refinement.py
#Programmer: Tim Tyree
#Date: 9.8.2022
import pandas as pd, numpy as np
from ...measures.auc import comp_auc_simple
from ...measures.point_process_measures import extract_fr_array_for_neuron

def gener_normal_random_perturbations(
        num_iter=100,
        random_state=42,
        use_random_seed=True,
        use_include_original=True,
    **kwargs):
    """default values of loc=0.0, scale=1.0, are can be modified through kwargs

    Example Usage:
rand_values1,rand_values2=gener_normal_random_perturbations(num_iter=100,random_state=42,use_random_seed=True)#,use_include_original=True,**kwargs)
    """
    if use_random_seed:
        np.random.seed(random_state)
    rand_values1=np.random.normal(size=num_iter+1, **kwargs)
    rand_values2=np.random.normal(size=num_iter+1, **kwargs)
    if use_include_original:
        #include the original at the start
        rand_values1[0]*=0.
        rand_values2[0]*=0.
    return rand_values1,rand_values2

def refine_candidate_tbins_for_neuron(df_tbins,rand_values1,rand_values2,booT,booF,spike_time_values_neuron,boo_cv1,boo_cv2,boo_cv3,
            perturbation_scale_factor=1.,
            min_dur=0.01,
            printing=True,**kwargs):
    """refine_candidate_tbins_for_neuron returns df_tbins  after performing refinement on it
    parallelizable method of predictive refinement. estimated run time ~2 minutes per 200 perturbations of ~100 tbins from one neuron.
    spike times for the neurons are encoded in spike_time_values_neuron.
    note on run time: computing fr_array is the slowest step. do it once by putting it at the end of the prediction generator.

    Parameters Settings
    --------------------
        perturbation_scale_factor: factor by which we multiply duration to get the stddev of perturbations for this tbin

        min_dur: seconds in the smallest candidate time bin before refinement

    Returns
    -------
    pandas.DataFrame
        df_tbins_refined: table of candidate with refinement

    Example Usage
    --------------
df_tbins_refined = refine_candidate_tbins_for_neuron(df_tbins,rand_values1,rand_values2,booT,booF,spike_time_values_neuron,boo_cv1,boo_cv2,boo_cv3,
            perturbation_scale_factor=1., #factor by which we multiply duration to get the stddev of perturbations for this tbin
            min_dur=0.01,printing=True)#,**kwargs)
    """
    # parallelizable method of predictive refinement
    if printing:
        print(f"found {df_tbins.shape[0]} time intervals that supported consistent auc sign convention according to 3-fold stratified cross-validation of the training trials.")

    #for each tbin with consistent sign, peform new predictive refinement method
    if printing:
        print(f"performing predictive refinement (estimated run time ~2 minutes for ~200 tbins)...")

    #format ground truth labels
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)
    label_values=np.concatenate((np.zeros(num_trialsF),np.ones(num_trialsT)))

    #compute auc by fold for each neuron
    df_tbins['refined']=0
    dict_refined_lst=[]
    for i_row,row in df_tbins.iterrows():
        tau1=row['tau1']
        tau2=row['tau2']
        sign=row['sign']
        #perform many perturbations for this tbin using the precomputed random values
        df_perturbations,fr_arrayT_,fr_arrayF_=gener_random_tbin_perturbations(tau1, tau2, sign, rand_values1, rand_values2,
                                    booT,booF,spike_time_values_neuron,
                                    min_dur=min_dur,perturbation_scale_factor=perturbation_scale_factor, **kwargs)

        #compute auc for the given 3fold crossval
        num_tintervals=df_perturbations.shape[0]
        auc_values_cv_=np.ones(shape=(num_tintervals,3))
        for j in range(num_tintervals):
            predictor_values_=sign*np.concatenate((fr_arrayF_[j],fr_arrayT_[j]))
            auc1_=comp_auc_simple(label_values[boo_cv1], predictor_values_[boo_cv1])
            auc2_=comp_auc_simple(label_values[boo_cv2], predictor_values_[boo_cv2])
            auc3_=comp_auc_simple(label_values[boo_cv3], predictor_values_[boo_cv3])
            auc_values_cv_[j,0]=auc1_
            auc_values_cv_[j,1]=auc2_
            auc_values_cv_[j,2]=auc3_
        df_perturbations['auc1']=auc_values_cv_[:,0]
        df_perturbations['auc2']=auc_values_cv_[:,1]
        df_perturbations['auc3']=auc_values_cv_[:,2]
        #identify any perturbation that satisfies a non-decreasing crossvalidation for all folds
        boo_non_worsened_cv =(df_perturbations['auc1']-row['auc1'])>=0
        boo_non_worsened_cv&=(df_perturbations['auc2']-row['auc2'])>=0
        boo_non_worsened_cv&=(df_perturbations['auc3']-row['auc3'])>=0
        boo_non_worsened_cv&=(df_perturbations['auc']-row['auc'])>=0
        if boo_non_worsened_cv.any():
            #keep the best such perturbation
            row_refined=df_perturbations[boo_non_worsened_cv].sort_values(by='auc',ascending=False).iloc[0]
            row_refined['refined']=1
            dict_refined_lst.append(dict(row_refined))
        else:
            dict_refined_lst.append(dict(row))
    df_tbins_refined=pd.DataFrame(dict_refined_lst)
    df_tbins_refined.sort_values(by='auc',ascending=False,inplace=True)
    return df_tbins_refined

def gener_random_tbin_perturbations(tau1, tau2, sign, rand_values1, rand_values2,
                            booT,booF,spike_time_values_neuron,
                            min_dur=0.01,
                            perturbation_scale_factor=0.5,
                            use_time_constraints=True,
                            taumin=0.,
                            taumax=3.6,
                             **kwargs):
    """
    Example Usage:
rand_values1,rand_values2=gener_random_normal_perturbations(num_iter=100,random_state=42,use_random_seed=True)#,use_include_original=True,**kwargs)
df_perturbations,fr_arrayT,fr_arrayF=gener_random_tbin_perturbations(tau1, tau2, sign, rand_values1, rand_values2,
                            booT,booF,spike_time_values_neuron,
                            min_dur=0.01,perturbation_scale_factor=0.5)#, **kwargs)
    """
    #compute perturbations for this tbin
    perturbation_scale=perturbation_scale_factor * (tau2-tau1)
    dtau1_values=perturbation_scale*rand_values1
    dtau2_values=perturbation_scale*rand_values2
    tau1_values=tau1+dtau1_values
    tau2_values=tau2+dtau2_values
    #remove any intervals with negative duration or less than thresh
    boo_dur_invalid=(tau2_values-tau1_values)<min_dur
    if use_time_constraints:
        boo_dur_invalid |= tau1_values<taumin
        boo_dur_invalid |= tau2_values>taumax
    tau1_values=tau1_values[~boo_dur_invalid].copy()
    tau2_values=tau2_values[~boo_dur_invalid].copy()
    #extract firing rate array for each perturbation
    fr_arrayT,fr_arrayF=extract_fr_array_for_neuron(tau1_values,tau2_values,booT,booF,spike_time_values_neuron)
    #format ground truth labels
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)
    label_values=np.concatenate((np.zeros(num_trialsF),np.ones(num_trialsT)))
    #preallocate memory
    num_tintervals=tau1_values.shape[0]
    auc_values=np.ones(num_tintervals)
    #compute auc for each perturbation
    for j in range(num_tintervals):
        predictor_values=sign*np.concatenate((fr_arrayF[j],fr_arrayT[j]))
        auc=comp_auc_simple(label_values, predictor_values)
        auc_values[j]=auc
    df_perturbations=pd.DataFrame({
        # 'nid':nid, # added later. nid is not in the local namespace.
        'tau1':tau1_values.copy(),
        'tau2':tau2_values.copy(),
        'auc':auc_values,
        'sign':sign
    })
    return df_perturbations,fr_arrayT,fr_arrayF
