# train_test_split_multimodal.py
#Programmer: Tim Tyree
#Date: 9.3.2022
from sklearn.model_selection import StratifiedKFold
import pandas as pd, numpy as np

def train_test_split_multimodal_crossval(concept_name_values_selected,d_labels,trial_num_values_remove,
                                         n_splits=5,
                                         shuffle=True,
                                         random_state=42,**kwargs):
    """train_test_split_multimodal_crossval returns d_labels with integer fields of the form
    'cv_fold_{mode}' indicating stratified crossvalidation folds where the trial is in the testing set.
    d_labels is a pandas.DataFrame instance with binary integer fields,
    'y_xmod' and 'y_MvMM' to indicate crossmodal (xmod) and match vs. mismatch (MvMM) trials, respectively.
    d_labels indicates identies by 'faceName', 'pheeName'.
    d_labels is uniequely indexed by trial.
    trial_num_values_remove are removed explicitely.

    Example Usage:
d_labels = train_test_split_multimodal_crossval(concept_name_values_selected,d_labels,trial_num_values_remove,
                                         n_splits=5,shuffle=True,random_state=42)
    """
    #identify unreasonable trials for removal
    boo_reasonable = d_labels['y_xmod']==d_labels['y_xmod'] # tautologically true
    for trial_num in trial_num_values_remove:
        boo_reasonable[trial_num] = False

    boo_xmod =d_labels['y_xmod']==1
    boo_MvMM =d_labels['y_MvMM']==1

    #identify selected identity match trials and label them by concept
    trial_num_values_face_only_lst=[]
    concept_labels_face_only_lst=[]
    trial_num_values_voice_only_lst=[]
    concept_labels_voice_only_lst=[]
    trial_num_values_match_lst=[]
    concept_labels_match_lst=[]
    trial_num_values_mismatch_lst=[]
    concept_labels_mismatch_lst=[]
    concept_label=0
    for concept_name in concept_name_values_selected:
        concept_label+=1
        boo_face =d_labels['faceName']==concept_name
        boo_voice=d_labels['pheeName']==concept_name
        #identify face-only trials
        trial_num_values_face_only = np.argwhere (( boo_reasonable & boo_face & ~boo_xmod).values).flatten()
        trial_num_values_face_only_lst.append(trial_num_values_face_only)
        concept_labels_face_only_lst.append(concept_label + np.zeros_like(trial_num_values_face_only))
        #identify voice-only trials
        trial_num_values_voice_only = np.argwhere (( boo_reasonable & boo_voice & ~boo_xmod).values).flatten()
        trial_num_values_voice_only_lst.append(trial_num_values_voice_only)
        concept_labels_voice_only_lst.append(concept_label + np.zeros_like(trial_num_values_voice_only))
        #identify identity match trials
        trial_num_values_match = np.argwhere (( boo_reasonable & boo_face & boo_voice ).values).flatten()
        trial_num_values_match_lst.append(trial_num_values_match)
        concept_labels_match_lst.append(concept_label + np.zeros_like(trial_num_values_match))
        #identify identity-mismatch trials
        trial_num_values_mismatch = np.argwhere (( boo_reasonable & (boo_face|boo_voice) & ~(boo_face&boo_voice) & boo_xmod).values).flatten()
        trial_num_values_mismatch_lst.append(trial_num_values_mismatch)
        concept_labels_mismatch_lst.append(concept_label + np.zeros_like(trial_num_values_mismatch))
        #test that the modes do not intersect
        assert np.intersect1d(trial_num_values_mismatch,trial_num_values_match).shape[0]==0
        assert np.intersect1d(trial_num_values_match,trial_num_values_face_only).shape[0]==0
        assert np.intersect1d(trial_num_values_voice_only,trial_num_values_face_only).shape[0]==0

    #perform stratified cross validation in all trials involving the selected concepts
    skf = StratifiedKFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)

    #perform stratified cross-validation of face-only trials (used for individual recognition predictions)
    trial_num_values_face_only=np.concatenate(trial_num_values_face_only_lst)
    y_face_only=np.concatenate(concept_labels_face_only_lst)
    d_labels['cv_fold_face_only']=-1
    cv_iter=0
    for train_index, test_index in skf.split(trial_num_values_face_only, y_face_only):
        cv_iter+=1
        trial_num_values_face_only_train = trial_num_values_face_only[train_index]
        trial_num_values_face_only_test  = trial_num_values_face_only[test_index]
        d_labels.loc[trial_num_values_face_only_test,'cv_fold_face_only']=cv_iter-1

    #perform stratified cross-validation of voice-only trials (used for individual recognition predictions)
    trial_num_values_voice_only=np.concatenate(trial_num_values_voice_only_lst)
    y_voice_only=np.concatenate(concept_labels_voice_only_lst)
    d_labels['cv_fold_voice_only']=-1
    cv_iter=0
    for train_index, test_index in skf.split(trial_num_values_voice_only, y_voice_only):
        cv_iter+=1
        trial_num_values_voice_only_train = trial_num_values_voice_only[train_index]
        trial_num_values_voice_only_test  = trial_num_values_voice_only[test_index]
        d_labels.loc[trial_num_values_voice_only_test,'cv_fold_voice_only']=cv_iter-1

    #perform stratified cross-validation of identity match trials (used for multiconcept recognition predictions)
    trial_num_values_match=np.concatenate(trial_num_values_match_lst)
    y_inm=np.concatenate(concept_labels_match_lst)
    d_labels['cv_fold_match']=-1
    cv_iter=0
    for train_index, test_index in skf.split(trial_num_values_match, y_inm):
        cv_iter+=1
        trial_num_values_inm_train = trial_num_values_match[train_index]
        trial_num_values_inm_test  = trial_num_values_match[test_index]
        d_labels.loc[trial_num_values_inm_test,'cv_fold_match']=cv_iter-1

    #uniquely label each combination of individuals involved in mismatch trials
    trial_num_values_mismatch=np.concatenate(trial_num_values_mismatch_lst)
    y_mm=np.concatenate(concept_labels_mismatch_lst)
    #stratify mismatch by (concept_label_lesser+1)**(concept_label_greater+1) bc unique label is made for each pair
    trial_num_set_values_mismatch=np.unique(trial_num_values_mismatch)
    concept_label_set_mismatch_lst=[]
    for trial_num in trial_num_set_values_mismatch:
        label_pair=y_mm[trial_num_values_mismatch==trial_num]
        ylow=np.min(label_pair)+1
        yhigh=np.max(label_pair)+1
        if ylow == yhigh: #if only one selected individual was involved,
            #then the label is the label,
            concept_label_set_mismatch_lst.append(ylow)
        else: #otherwise, define a new, unique label
            #NOTE: if stratification isn't working for mismatch, consider using least common of ylow,yhigh to update
            concept_label_set_mismatch_lst.append(-1*(yhigh*ylow+1))

    #perform stratified cross-validation of identity mismatch trials (used for inm recognition predictions)
    # concept_label_set_values_mismatch=np.array(concept_label_set_mismatch_lst)
    y_mismatch=np.array(concept_label_set_mismatch_lst)
    d_labels['cv_fold_mismatch']=-1
    cv_iter=0
    for train_index, test_index in skf.split(trial_num_set_values_mismatch, y_mismatch):
        cv_iter+=1
        trial_num_values_mismatch_train = trial_num_values_mismatch[train_index]
        trial_num_values_mismatch_test  = trial_num_values_mismatch[test_index]
        d_labels.loc[trial_num_values_mismatch_test,'cv_fold_mismatch']=cv_iter-1

    #test that the modes do not intersect
    assert np.intersect1d(trial_num_set_values_mismatch,trial_num_values_match).shape[0]==0
    assert np.intersect1d(trial_num_values_match,trial_num_values_face_only).shape[0]==0
    assert np.intersect1d(trial_num_values_voice_only,trial_num_values_face_only).shape[0]==0
    return d_labels
