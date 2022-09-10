#cross_validation.py
#Programmer: Tim Tyree
#Date: 6.2.2022
import numpy as np

def gener_boolean_3fold_split(num_trials,random_state=42):
    """gener_boolean_3fold_split randomly generates 3fold random split and returns the result as a boolean array with num_trials entries.
    gener_boolean_3fold_split seeds the numpy random number generator to random_state
    gener_boolean_3fold_split does the following at a high level:
    - randomly select 1/3 of true trials as boo_cv_1
    - randomly select 1/2 of remaining true trials as boo_cv_2
    - randomly select 1/3 of false trials
    - randomly select 1/2 of remaining false trials as boo_cv_2
    - any remaining trials are placed in boo_cv3

    Example Usage:
boo_cv1_valuesF,boo_cv2_valuesF,boo_cv3_valuesF=gener_boolean_3fold_split(num_trialsF,random_state=42)
assert (boo_cv1_valuesF|boo_cv2_valuesF|boo_cv3_valuesF).all()
assert not (boo_cv2_valuesF&boo_cv3_valuesF).any()
assert not (boo_cv1_valuesF&boo_cv3_valuesF).any()
assert not (boo_cv1_valuesF&boo_cv2_valuesF).any()
    """
    num_trialsF=int(num_trials)
    np.random.seed(int(random_state))
    trial_index_valuesF=np.arange(num_trialsF)
    boo_cv1_valuesF=trial_index_valuesF<0
    boo_cv2_valuesF=trial_index_valuesF<0
    trial_index_valuesF_cv1 = np.random.choice(trial_index_valuesF, size=int(num_trialsF/3), replace=False)
    trial_index_valuesF_not_cv1=np.setdiff1d(trial_index_valuesF,trial_index_valuesF_cv1)
    trial_index_valuesF_cv2 = np.random.choice(trial_index_valuesF_not_cv1, size=int(num_trialsF/3), replace=False)
    boo_cv1_valuesF[trial_index_valuesF_cv1]=True
    boo_cv2_valuesF[trial_index_valuesF_cv2]=True
    boo_cv3_valuesF=(~boo_cv1_valuesF)&(~boo_cv2_valuesF)
    return boo_cv1_valuesF,boo_cv2_valuesF,boo_cv3_valuesF
