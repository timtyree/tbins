#counter.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import pandas as pd, numpy as np

def count_abundance_concepts(df_labels,boot,printing=False,**kwargs):
    """select_abundant_concepts counts the number of trials
    that each concept was presented and then returns
    that information as a pandas.DataFrame instance.
    boot is a boolean index indicating trials to be removed
    from consideration because they dmeonstrated an obvious recording error.

    Example Usage:
df_concept_count = count_abundance_concepts(df_labels,boot,printing=True)
    """
    #get list of individuals in alphabetical order
    concept_name_set = set(df_labels['pheeName'].drop_duplicates())
    concept_name_set.update(set(df_labels['faceName'].drop_duplicates()))
    concept_name_set.remove('none')
    concept_name_lst = sorted(concept_name_set)
    if printing:
        print(f"all individuals presented:\n\t\t{concept_name_lst}")

    boo_xmod =df_labels['y_xmod']==1
    boo_MvMM =df_labels['y_MvMM']==1
    dict_concept_count_lst=[]
    for concept_name in concept_name_lst:
        boo_face =df_labels['faceName']==concept_name
        boo_voice=df_labels['pheeName']==concept_name
        num_appearances =( ~boot & (boo_face|boo_voice)).sum()
        num_face_only=( ~boot & boo_face & ~boo_xmod).sum()
        num_voice_only=( ~boot & boo_voice & ~boo_xmod).sum()
        num_match       =( ~boot & (boo_face&boo_voice)).sum()
        num_mismatch       =( ~boot & (boo_face|boo_voice) & ~(boo_face&boo_voice) & boo_xmod).sum()
        num_mismatch_overall       =( ~boot & ~boo_MvMM & boo_xmod).sum()

        assert num_appearances== num_face_only+num_voice_only+num_match+num_mismatch

        #record
        dict_concept_count = dict(concept_name=concept_name,
            num_appearances=num_appearances,
            num_face_only=num_face_only,
            num_voice_only=num_voice_only,
            num_match=num_match,
            num_mismatch=num_mismatch,
            num_mismatch_overall=num_mismatch_overall,)
        dict_concept_count_lst.append(dict_concept_count)
    df_concept_count = pd.DataFrame(dict_concept_count_lst)
    return df_concept_count
