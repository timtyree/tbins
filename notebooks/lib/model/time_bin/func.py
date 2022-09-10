# time_bin.func.py
#Programmer: Tim Tyree
#Date: 9.8.2022
import pandas as pd, numpy as np
from ...measures.auc import comp_auc_simple
from ...measures.point_process_measures import extract_fr_array_for_neuron

def gener_candidate_tbins_for_neuron_fast(tau1_values,tau2_values,booT,booF,spike_time_values_neuron,random_state=42,**kwargs):
    """gener_candidate_tbins_for_neuron_fast returns a pandas.DataFrame instance with rows indicating candidate time bins as df_tbins_weak
    gener_candidate_tbins_for_neuron_fast performs 3fold stratified crossvalidation on the training trials to detect
    a general ability to distinguish true trials from false trials.

    df_tbins_weak has the following fields:
    - tau1: start time of time bin
    - tau2: end time of time bin
    - sign: +/-1 to indicate a true/false dominant firing rate response, respectively.
    - auc1,2,3: indicate the area under the roc curve for each fold, respectively.

    Example Usage:
df_tbins_weak=gener_candidate_tbins_for_neuron_fast(tau1_values,tau2_values,booT,booF,spike_time_values_neuron,random_state=42)#,**kwargs)
    """
    num_tintervals=tau1_values.shape[0]
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)

    # extract_fr_array_for_neuron
    fr_arrayT,fr_arrayF=extract_fr_array_for_neuron(tau1_values,tau2_values,booT,booF,spike_time_values_neuron)
    #format ground truth labels
    label_values=np.concatenate((np.zeros(num_trialsF),np.ones(num_trialsT)))

    #compute the overall sign that supports a larger auc
    #preallocate memory often rewritten to often
    sign_values=np.ones(num_tintervals)
    auc_values=np.ones(num_tintervals)
    for j in range(num_tintervals):
        predictor_values=np.concatenate((fr_arrayF[j],fr_arrayT[j]))
        auc=comp_auc_simple(label_values, predictor_values)
        if auc>=0.5:
            auc_values[j]=auc
        else:
            auc_values[j]=1.-auc
            sign_values[j]*=-1

    df_gridsearch=pd.DataFrame({
    #     'nid':nid,
        'tau1':tau1_values.copy(),
        'tau2':tau2_values.copy(),
        'auc':auc_values,
        'sign':sign_values
    })

    num_trialsT=sum(booT)
    num_trialsF=sum(booF)
    #generate 3-fold stratified cross-validation
    boo_cv1F,boo_cv2F,boo_cv3F=gener_boolean_3fold_crossval_split(num_trialsF,random_state=random_state)
    boo_cv1T,boo_cv2T,boo_cv3T=gener_boolean_3fold_crossval_split(num_trialsT,random_state=random_state)
    boo_cv1=np.concatenate((boo_cv1F,boo_cv1T))
    boo_cv2=np.concatenate((boo_cv2F,boo_cv2T))
    boo_cv3=np.concatenate((boo_cv3F,boo_cv3T))

    #compute the overall sign that supports a larger auc
    sign_values_cv=np.ones(shape=(num_tintervals,3))
    auc_values_cv=np.ones(shape=(num_tintervals,3))
    for j in range(num_tintervals):
        predictor_values=np.concatenate((fr_arrayF[j],fr_arrayT[j]))
        auc1=comp_auc_simple(label_values[boo_cv1], predictor_values[boo_cv1])
        auc2=comp_auc_simple(label_values[boo_cv2], predictor_values[boo_cv2])
        auc3=comp_auc_simple(label_values[boo_cv3], predictor_values[boo_cv3])
        if auc1>=0.5:
            auc_values_cv[j,0]=auc1
        else:
            auc_values_cv[j,0]=1.-auc1
            sign_values_cv[j,0]*=-1
        if auc2>=0.5:
            auc_values_cv[j,1]=auc2
        else:
            auc_values_cv[j,1]=1.-auc2
            sign_values_cv[j,1]*=-1
        if auc3>=0.5:
            auc_values_cv[j,2]=auc3
        else:
            auc_values_cv[j,2]=1.-auc3
            sign_values_cv[j,2]*=-1

    #determine which tbins of these have consistent sign with overall in addition to all three folds of cross-validation
    tinterval_index_lst_same_sign_consistent=[]
    for j in range(num_tintervals):
        if np.abs(np.sum(sign_values_cv[j]) + sign_values[j])>3:
            # only if all three signs are the same value
            tinterval_index_lst_same_sign_consistent.append(j)
    #len(tinterval_index_lst_same_sign_consistent),num_tintervals#, np.max(auc_values_cv,axis=0)

    #filter possiblilities by consistency of sign values
    df_gridsearch['auc1']=auc_values_cv[:,0]
    df_gridsearch['auc2']=auc_values_cv[:,1]
    df_gridsearch['auc3']=auc_values_cv[:,2]
    # df_gridsearch['sign']=sign_values
    df_tbins_weak=df_gridsearch.loc[tinterval_index_lst_same_sign_consistent]
    return df_tbins_weak

def compute_interval_overlap(t1,t2,s1,s2):
    return np.min((t2,s2)) - np.max((t1,s1))

def select_nonoverlapping_tbins_for_neuron(df,priority_col='auc',
                                max_dur_overlap=0., #seconds #zero overlap assures disjoint tbins
                                **kwargs):
    """
    Example Usage:
df_tbins_neuron=select_nonoverlapping_tbins_for_neuron(df,max_dur_overlap=0.)
    """
    df_sorted=df.sort_values(by=priority_col,ascending=False)
    #generate list of refined time bins where the next tbin does not overlap with any previous tbin by more than a threshold amount
    index_lst=[]
    for i,row in df_sorted.iterrows():
        if len(index_lst)==0:
            index_lst.append(i)
        else:
            #check whether this row is shares less than a threshold duration with any other tbin
            boo_overlaps=False
            s1=row['tau1']
            s2=row['tau2']
            for ii in index_lst:
                t1=df_sorted.loc[ii,'tau1']
                t2=df_sorted.loc[ii,'tau2']
                dur_overlap=compute_interval_overlap(t1,t2,s1,s2)
                if max_dur_overlap<dur_overlap:
                    boo_overlaps=True
            if not boo_overlaps:
                index_lst.append(i)
    df_tbins_neuron = df_sorted.loc[index_lst]
    return df_tbins_neuron

def gener_boolean_3fold_crossval_split(num_trials,random_state=42):
    """gener_boolean_3fold_crossval_split returns a tuple of boolean 1D numpy.array instances that are true for the testing trials of each split.
    gener_boolean_3fold_crossval_split randomly generates 3fold random split and returns the result as a boolean array with num_trials entries.
    gener_boolean_3fold_crossval_split seeds the numpy random number generator to random_state
    gener_boolean_3fold_crossval_split does the following at a high level:
    - randomly select 1/3 of true trials as boo_cv_1
    - randomly select 1/2 of remaining true trials as boo_cv_2
    - randomly select 1/3 of false trials
    - randomly select 1/2 of remaining false trials as boo_cv_2
    - any remaining trials are placed in boo_cv3

    Example Usage:
boo_cv1_valuesF,boo_cv2_valuesF,boo_cv3_valuesF=gener_boolean_3fold_crossval_split(num_trialsF,random_state=42)
assert (boo_cv1_valuesF|boo_cv2_valuesF|boo_cv3_valuesF).all()
assert not (boo_cv2_valuesF&boo_cv3_valuesF).any()
assert not (boo_cv1_valuesF&boo_cv3_valuesF).any()
assert not (boo_cv1_valuesF&boo_cv2_valuesF).any()
    """
    num_trialsF=int(num_trials)
    np.random.seed(int(random_state))
    trial_index_valuesF=np.arange(num_trialsF)
    #initialize indices to false by default
    boo_cv1_valuesF=trial_index_valuesF<0
    boo_cv2_valuesF=trial_index_valuesF<0

    trial_index_valuesF_cv1 = np.random.choice(trial_index_valuesF, size=int(num_trialsF/3), replace=False)
    trial_index_valuesF_not_cv1=np.setdiff1d(trial_index_valuesF,trial_index_valuesF_cv1)
    trial_index_valuesF_cv2 = np.random.choice(trial_index_valuesF_not_cv1, size=int(num_trialsF/3), replace=False)
    boo_cv1_valuesF[trial_index_valuesF_cv1]=True
    boo_cv2_valuesF[trial_index_valuesF_cv2]=True
    boo_cv3_valuesF=(~boo_cv1_valuesF)&(~boo_cv2_valuesF)
    return boo_cv1_valuesF,boo_cv2_valuesF,boo_cv3_valuesF
