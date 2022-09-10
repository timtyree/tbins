#point_process_measures.py
#Programmer: Tim Tyree
#Date: 10.9.2021
import numpy as np,pandas as pd

#########################
# Primitive Rate Measures
#########################
def comp_fr(spike_time_values,tau1,tau2):
    boo=(spike_time_values>=tau1)&(spike_time_values<tau2)
    spike_count=np.sum(boo)
    FR=spike_count/(tau2-tau1) #Hz
    return FR

def comp_FR(spike_time_values,tau1,tau2):
    boo=(spike_time_values>=tau1)&(spike_time_values<tau2)
    spike_count=spike_time_values[boo].shape[0]
    FR=spike_count/(tau2-tau1) #Hz
    return FR

def comp_FR_lst(spike_time_values,startTimes,endTimes):
    """
    Example Usage:
FR_values=np.array(comp_FR_lst(spike_time_values[trial_num], startTimes=bins[:-1], endTimes=bins[1:]))
    """
    FR_lst=[]
    for tau1,tau2 in zip(startTimes,endTimes):
        boo=(spike_time_values>=tau1)&(spike_time_values<tau2)
        spike_count=float(spike_time_values[boo].shape[0])
        FR=spike_count/(tau2-tau1) #Hz
        FR_lst.append(FR)
    return FR_lst

#########################
# Firing Rate Arrays
#########################
def extract_simple_firing_rates(spike_time_array, df_tbins,tbin_cols=['nid','tau1','tau2','p'],**kwargs):
    """extract_simple_firing_rates supposes only one trial is represented in df_tbins.
    df_tbins has unique values for fields ['nid','tau1','tau2'] (can be modified by tbin_cols kwarg)
    spike_time_array is a numpy array with shape num_trials,num_neurons and entries that are 1D numpy arrays
    extract_simple_firing_rates returns
        - X, which is a numpy array with shape num_trials,num_tbins with entries of nonegative floats
        - tbin_indices, which is a numpy array with information on
    Example Usage:
X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins)
    """
    s = set(df_tbins.columns)
    #if undefined, set default p to be larger than unity
    if not s.issuperset({'p'}):
        df_tbins['p']=9999.

    num_trials=spike_time_array.shape[0]
    tbin_indices=df_tbins[tbin_cols].values
    X=np.zeros(shape=(num_trials,tbin_indices.shape[0]))*np.nan
    #compute firing rate features for each neuron in each trial
    tbin_indices_=tbin_indices[:,:3]
    for trial_num in range(num_trials):
        spike_time_array_=spike_time_array[trial_num]
        for i,(nid,tau1,tau2) in enumerate(tbin_indices_):
            spike_time_values=spike_time_array_[int(nid)]
            FR=comp_FR(spike_time_values,tau1,tau2)
            X[trial_num,i]=FR
    return X,tbin_indices

def extract_fr_array_for_neuron(tau1_values,tau2_values,booT,booF,spike_time_values_neuron):
    """
    Example Usage:
fr_arrayT,fr_arrayF=extract_fr_array_for_neuron(tau1_values,tau2_values,booT,booF,spike_time_values_neuron)
    """
    num_tintervals=tau1_values.shape[0]
    num_trialsT=sum(booT)
    num_trialsF=sum(booF)

    #extract the spike times from the true/false training trials
    spike_time_values_neuronT=spike_time_values_neuron[booT]
    spike_time_values_neuronF=spike_time_values_neuron[booF]

    #preallocate memory for holding firing rates
    fr_arrayT=np.zeros(shape=(num_tintervals,num_trialsT))
    fr_arrayF=np.zeros(shape=(num_tintervals,num_trialsF))

    #fill firing rate arrays with observed values
    for i in range(num_trialsT):
        spike_time_values=spike_time_values_neuronT[i]
        for j in range(num_tintervals):
            fr_arrayT[j,i]=comp_fr(spike_time_values,tau1_values[j],tau2_values[j])
    for i in range(num_trialsF):
        spike_time_values=spike_time_values_neuronF[i]
        for j in range(num_tintervals):
            fr_arrayF[j,i]=comp_fr(spike_time_values,tau1_values[j],tau2_values[j])
    return fr_arrayT,fr_arrayF

###########################
# Primitive Event Measures
###########################
def hindsight_delay(spike_time_values,t):
    boo=spike_time_values<t
    if not boo.any():
        return np.nan
    delay = t-np.max(spike_time_values[boo])
    return delay

def foresight_delay(spike_time_values,t):
    """"""
    boo=spike_time_values>t
    if not boo.any():
        return np.nan
    forward_delay = np.min(spike_time_values[boo])-t
    return forward_delay

@np.vectorize
def signed_connection_rate(tau_minus,tau_plus,tau0=1.,**kwargs):
    """signed_connection_rate returns the signed connection rate
    from the hindsight delay, tau_minus, to the foresight delay, tau_plus.
    tau_minus,tau_plus,tau0=1. are nonnegative float primitives
    tau_minus,tau_plus are primal event delay measures.
    tau0 is a characteristic timescale, defaulting to unit time.

    Example Usage:
c=signed_connection_rate(tau_minus,tau_plus)#,tau0=1.)
    """
    if np.isnan(tau_minus)&np.isnan(tau_plus):
        return 0.
    if (tau_minus==0)|(tau_plus==0):#edge case shouldn't happen by earlier consruction of the primal event delay measures
        return 0.
    if np.isnan(tau_minus):
        return tau0/tau_plus
    if np.isnan(tau_plus):
        return -tau0/tau_minus
    if tau_minus>tau_plus:
        return tau0/tau_plus
    return -tau0/tau_minus
