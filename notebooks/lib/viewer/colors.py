#colors.py
import matplotlib.pyplot as plt, numpy as np, pandas as pd

def get_cmap(n, name='Paired'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.

    Example Usage:
fig,ax=plt.subplots(figsize=(6,4))
cmap = get_cmap(len(fn_lst_displacements))
for i, fn in enumerate(fn_lst_displacements):
    dict_dense_umap_displacements=load_from_pkl(fn)
    color=cmap(i)
    '''
    return plt.cm.get_cmap(name, n)

def compute_colors_for_individuals(session_num,concept_name_to_label_dict,cmap_name='Paired',**kwargs):
    """color scheme used in the main text for alphabetically ordered conspecifics.
    concep_names come from the keys of concept_name_to_label_dict.

    these are a couple reasonable alternatives to cmap_name:
        cmap_name='viridis'
        cmap_name='hsv'

    Example Usage:
label_to_concept_name_dict,concept_name_to_color_dict=compute_colors_for_individuals(session_num,concept_name_to_label_dict,cmap_name='Paired')
    """
    #map from concept_name to label and vice versa
    concept_name_basis_values=np.array(list(concept_name_to_label_dict.keys()))
    label_basis_values=np.array(list(concept_name_to_label_dict.values()))
    label_to_concept_name_dict=dict(zip(label_basis_values,concept_name_basis_values))

    #generate dictionary map from concept_name to integer value, unique color
    cmap = get_cmap(concept_name_basis_values.shape[0]+1,name=cmap_name)
    color_basis_lst=[]
    for i, concept_name in enumerate(concept_name_basis_values):
        color=cmap(i)
        color_basis_lst.append(color)

    #map from concept_name to color and vice versa
    concept_name_to_color_dict=dict(zip(concept_name_basis_values,color_basis_lst))
    color_to_concept_name_dict=dict(zip(color_basis_lst,concept_name_basis_values))
    return label_to_concept_name_dict,concept_name_to_color_dict
