# time_bins.time_bins_main.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import pandas as pd, numpy as np
from .func import gener_candidate_tbins_for_neuron_fast,gener_boolean_3fold_crossval_split,select_nonoverlapping_tbins_for_neuron
from .refinement import refine_candidate_tbins_for_neuron,gener_normal_random_perturbations
from ...measures.distinguishability import estimate_distinguishability_tbins
from ...measures.point_process_measures import extract_simple_firing_rates
from ...utils.progress_bar import printProgressBar

def gener_tbins_fast(spike_time_array,booT,booF,
    nid_values=None,
    taumin=0,
    taumax=3.6,
    delta_tau_min=0.2,
    refinement=True,
    num_perturbations=100,
    num_cv_samples=5,
    random_state=42,
    use_random_seed=True,
    perturbation_scale_factor=1.,
    min_dur=0.2,
    max_dur_overlap=9999.,
    printing=False,**kwargs):
    """gener_tbins_fast returns a tuple of df_tbins and df_tbins_refined, respectively.
    booT,booF are boolean index arrays indexing the true/false training trials, respectively.
    spike_time_array is a 2D numpy array instance of list objects that contain spike times for a given trial-neuron pair.
    decreasing max_dur_overlap may needlessly remove useful predictive time bins,
    so its default value is set arbitrarily large while remaining small enough to be a float32 instance.
    using refinement may add ~2minutes to the estimated run time (default: refinement=True).
    otherwise, gener_tbins_fast can run in tpyically less than 2 minutes per call.

    Parameters Settings
    --------------------
        nid_values: neuron index values to consider.  all neurons are considered if nid_values is None (default: nid_values=None)

        taumin: earliest start time

        taumax: latest end time

        delta_tau_min: time between two start/end times

        refinement: whetehr or not to use refinement

        delta_tau_min: seconds between each division of time

        num_perturbations: number of random perturbations

        perturbation_scale_factor: factor by which we multiply duration to get the stddev of perturbations for this tbin

        min_dur: seconds in the smallest candidate time bin before refinement

        max_dur_overlap: seconds of the maximum allowed overlap between candidate time bin _after_ refinement.

    Returns
    -------
    tuple
        df_tbins,df_tbins_refined: table of candidate time bins without and with refinement, repectively.  df_tbins_refined is None if refinement is False.


    Example Usage
    --------------
X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins)
df_tbins_cv,df_tbins_refined_cv =  gener_tbins_fast(spike_time_array,booT,booF,refinement=True,num_cv_samples=5,printing=True)#**kwargs
    """
    num_iter = num_perturbations
    tau_values=np.arange(taumin,taumax+delta_tau_min,delta_tau_min)
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)
    if printing:
        num_tpartitions = tau_values.shape[0]-1
        print(f"considering {num_tpartitions} equal partitions from {taumin=} to {taumax=} ({num_trialsT=}, {num_trialsF=})...")

    #create array for grid search of time intervals
    tau1_lst=[]
    tau2_lst=[]
    for tau1 in tau_values[:-1]:
        for tau2 in tau_values[tau_values>tau1]:
            tau1_lst.append(tau1)
            tau2_lst.append(tau2)
    tau1_values=np.array(tau1_lst)
    tau2_values=np.array(tau2_lst)

    num_tintervals=tau1_values.shape[0]
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)
    #generate 3-fold stratified cross-validation
    boo_cv1F,boo_cv2F,boo_cv3F=gener_boolean_3fold_crossval_split(num_trialsF,random_state=random_state)
    boo_cv1T,boo_cv2T,boo_cv3T=gener_boolean_3fold_crossval_split(num_trialsT,random_state=random_state)
    boo_cv1=np.concatenate((boo_cv1F,boo_cv1T))
    boo_cv2=np.concatenate((boo_cv2F,boo_cv2T))
    boo_cv3=np.concatenate((boo_cv3F,boo_cv3T))

    if nid_values is None:
        num_neurons=spike_time_array.shape[1]
        nid_values=np.array(list(range(num_neurons)))
    else:
        num_neurons=nid_values.shape[0]
    if printing:
        print(f"generating tbins for {num_neurons} neurons...")

    #generate tbins without refinement using the fast method based on cross-validated maximization of auc
    df_tbins_neuron_lst=[]
    step=0
    nsteps=num_neurons
    for nid in nid_values:
        spike_time_values_neuron=spike_time_array[:,nid]
        df_tbins_weak=gener_candidate_tbins_for_neuron_fast(tau1_values,tau2_values,booT,booF,spike_time_values_neuron,random_state=random_state)#,**kwargs)
        df_tbins_neuron=select_nonoverlapping_tbins_for_neuron(df_tbins_weak,max_dur_overlap=0.)
        df_tbins_neuron['nid']=nid
        #record
        df_tbins_neuron_lst.append(df_tbins_neuron)
        if printing:
            step+=1
            printProgressBar(step + 1, nsteps, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #estimate general ability to distinguish trials
    df_tbins=pd.concat(df_tbins_neuron_lst).copy()
    df_tbins.reset_index(inplace=True,drop=True)
    df_tbins['p']=9999.
    X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins)
    df_tbins_cv,dict_tbins_cv_lst=estimate_distinguishability_tbins(X,tbin_indices,booT,booF,n_splits=num_cv_samples)

    ##optionally, refine tbins
    if refinement:
        # gener_tbin_perturbations to be reused for each tbin (way faster than generating a bajillion normal samples)
        rand_values1,rand_values2=gener_normal_random_perturbations(num_iter=num_iter,random_state=random_state,use_random_seed=use_random_seed)#,use_include_original=True,**kwargs)
        df_tbins_neuron_refined_lst=[]
        step=0
        for df_tbins_neuron in df_tbins_neuron_lst:
            if df_tbins_neuron.shape[0]>0:
                nid=df_tbins_neuron['nid'].max()
                spike_time_values_neuron=spike_time_array[:,nid]
                df_tbins_refined = refine_candidate_tbins_for_neuron(df_tbins_neuron,rand_values1,rand_values2,booT,booF,spike_time_values_neuron,boo_cv1,boo_cv2,boo_cv3,
                            perturbation_scale_factor=perturbation_scale_factor,
                            min_dur=min_dur,printing=False)#,**kwargs)
                df_tbins_neuron_refined=select_nonoverlapping_tbins_for_neuron(df_tbins_refined,max_dur_overlap=max_dur_overlap)
                df_tbins_neuron_refined['nid']=nid
                df_tbins_neuron_refined_lst.append(df_tbins_neuron_refined)
                if printing:
                    step+=1
                    nsteps=num_neurons+1
                    printProgressBar(step + 1, nsteps, prefix = 'Progress:', suffix = 'Complete', length = 50)

        #generate tbins for all neurons
        df_tbins_refined=pd.concat(df_tbins_neuron_refined_lst).copy()
        df_tbins_refined.reset_index(inplace=True,drop=True)
        df_tbins_refined['p']=9999.
        X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins_refined)
        df_tbins_refined_cv,dict_tbins_refined_cv_lst=estimate_distinguishability_tbins(X,tbin_indices,booT,booF,n_splits=num_cv_samples)
    else:
        df_tbins_refined_cv=None
    return df_tbins_cv,df_tbins_refined_cv
