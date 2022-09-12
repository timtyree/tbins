# recording_warnings.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import pandas as pd, numpy as np

def count_streak(boor):
    """
    Example Usage:
longest_error_streak,final_error_streak = count_streak(boor)
    """
    longest_error_streak=0
    final_error_streak=0
    found_non_error=False
    error_streak=0
    for boo in np.flip(boor):
        if not boo: #error has not occurred
            found_non_error=True
            error_streak=0 # reset counter
        if not found_non_error:
            final_error_streak+=1
        if boo: #if an error has occurred, count
            error_streak+=1
            if error_streak>longest_error_streak:
                longest_error_streak=error_streak
    return longest_error_streak,final_error_streak

def count_spike_count_errs(spike_time_array,
                           min_refractory_period = 1/150, #150 Hz max firing rate
                           **kwargs):
    """count_spike_count_errs identifies the indices of obvious recording errors.
    max FR is 1/min_refractory_period. min_refractory_period is in seconds..
    error code 3 may trigger for multiunits with meaningful information.
    error_codes: {0:'No Error', 1:'No Spikes', 2:'One Spike', 3:'Exceeds Max FR'}

    Example Usage:
spike_count_array,error_code_array = count_spike_count_errs(spike_time_array,min_refractory_period = 1/150)#, #150 Hz max firing rate
    """
    spike_count_array = np.zeros_like(spike_time_array,dtype=int)
    error_code_array = np.zeros_like(spike_time_array,dtype=int)
    num_trials, num_neurons = spike_time_array.shape
    for trial_num in range(num_trials):
        for nid in range(num_neurons):
            x = np.array(spike_time_array[trial_num,nid])
            if not x.dtype == float:
                error_code_array[trial_num,nid]=1 #is not float...  likely bc monkey moved or the neuron died...
            else:
                #cast any single spikes to 1D arrays
                if len(x.shape)==0:
                    x = np.array([x])
                    error_code_array[trial_num,nid]=2 #one spike per trial could be valid...
                    spike_count_array[trial_num,nid]=1
                elif x.shape[0]==1:
                    error_code_array[trial_num,nid]=2 #one spike per trial could be valid...
                    spike_count_array[trial_num,nid]=1
                else:
                    spike_count_array[trial_num,nid]=x.shape[0]
                    min_delay= np.min(np.diff(x))
                    if min_delay < min_refractory_period:
                        error_code_array[trial_num,nid]=3 #is likely a multi-unit...
    return spike_count_array,error_code_array

def identify_obvious_recording_warnings_from_session(error_code_array,spike_count_array,
                                                 max_count_err_3_by_trial=75,
                                                 max_count_err_3_by_neuron=350,
                                                 sort=False,
                                                 printing=True,
                                                 **kwargs):
    """identify_obvious_recording_warnings_from_session counts streaks in error code 1 which are returned as a pandas.DataFrame instance

    Example Usage:
spike_count_array,error_code_array = count_spike_count_errs(spike_time_array,min_refractory_period = 1/150)#, #150 Hz max firing rate
error_code=3
#choose filter for trials that are unreasonable (error_code==3 too often)
count_err_3_by_trial = np.sum(error_code_array==error_code,axis=1)
#choose filter for neurons that are unreasonable (error_code==3 too often)
count_err_3_by_neuron = np.sum(error_code_array==error_code,axis=0)
#boolean filter for error code 3 count being above threshold
boot3 = count_err_3_by_trial >max_count_err_3_by_trial
boon3 = count_err_3_by_neuron>max_count_err_3_by_neuron
num_errorcode3_violations_neurons = sum(boon3)
num_errorcode3_violations_trials = sum(boot3)
if printing:
    print(f"num. errors: {num_errorcode3_violations_neurons} out of {boon3.shape[0]} neurons")
    print(f"num. errors: {num_errorcode3_violations_trials} out of {boot3.shape[0]} trials")

df_es = identify_obvious_recording_warnings_from_session(error_code_array,spike_count_array,
                                                 max_count_err_3_by_trial=75,
                                                 max_count_err_3_by_neuron=350,
                                                 printing=True,**kwargs)
    """
    num_neurons = error_code_array.shape[1]
    boo_err  = error_code_array==1
    boo_err |= error_code_array==2
    #choose filter for trials that are unreasonable
    count_err_1_by_trial = np.sum(boo_err,axis=1)
    #choose filter for neurons that are unreasonable
    count_err_1_by_neuron = np.sum(boo_err,axis=0)
    #count error streaks
    booerr = (boo_err).T
    longest_error_streak_lst=[]
    final_error_streak_lst=[]
    max_spike_count_lst=[]
    for nid in range(num_neurons):
        boor = booerr[nid].copy()
        #identify trials with consecutive error
        longest_error_streak,final_error_streak = count_streak(boor)
        #record longest_error_streak,final_error_streak
        longest_error_streak_lst.append(longest_error_streak)
        final_error_streak_lst.append(final_error_streak)
        #compute max num spikes per trial
        max_spike_count = np.max(spike_count_array.T[nid])
        #record
        max_spike_count_lst.append(max_spike_count)

    #choose filter for neurons that are unreasonable (error code == 1 too often)
    df_es = pd.DataFrame(dict(nid=list(range(num_neurons)),
                              longest_error_streak=longest_error_streak_lst,
                              final_error_streak=final_error_streak_lst,
                              max_spike_count=max_spike_count_lst,
                             ))
    df_es.sort_values(by=['final_error_streak','longest_error_streak'],ascending=False)
    return df_es
