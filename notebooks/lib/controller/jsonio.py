# jsonio.py
#Programmer: Tim Tyree
#Date: 9.9.2022
import json, os, sys, re, numpy as np, pandas as pd

def write_parameters_to_json(param_dict, param_file_name):
    '''Example Usage:
    write_parameters_to_json(kwargs, 'lib/param_set_8.json')
    '''
    with open(param_file_name, 'w') as json_file:
        json.dump(param_dict, json_file)
    return True

def read_parameters_from_json(param_file_name):
    '''Example Usage:
    kwargs = read_parameters_from_json('lib/param_set_8.json')
    '''
    with open(param_file_name) as json_file:
        data = json.load(json_file)
        return data

#the only class i defined in tbins/*
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_dict_to_json(pdict,save_fn,indent=1,sort_keys=False):
    #save all output values as json
    os.system('touch '+save_fn)
    with open(save_fn,"w") as fp:
        json.dump(dict(pdict),fp,cls=NpEncoder,indent=indent,sort_keys=sort_keys)
    return True

def save_to_json(pdict,save_fn,**kwargs):
    #save all output values as json
    return save_dict_to_json(pdict,save_fn,**kwargs)

def load_from_json(file_name):
    '''
    Example Usage:
kwargs = read_parameters_from_json('lib/param_set_8.json')
    '''
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def convert_to_dict_recursive(dict_input,
                              type_ignore_set=set(["<class 'xgboost.sklearn.XGBClassifier'>"])):
    """convert_to_dict_recursive converts pandas data to json returns a json serializable version of dict_input

    convert_to_dict_recursive recursively searches nested dictionary, tuple, or list given as dict_input,
    and converts any pandas.DataFrame instances
    to the dictionary primative for serializability.
    Recursive search supports iteration over list and tuple iterables.

    TODO(later): generalize convert_to_dict_recursive to take any iterables...
      Q: Can lst.__iter__() be used with type()?
      NOTE: dict instances are iterable...

    Example Usage:
dict_output=convert_to_dict(dict_input)
    """
    if type_ignore_set.issuperset(set([str(type(dict_input))])):
        return 'Warning: this value removed for serializability by convert_to_dict_recursive'
    if type(dict_input)==type(pd.Series(dtype='float64')):
        dict_input=dict(dict_input)
    if type(dict_input)==type(pd.DataFrame()):
        #return convert_to_dict_recursive(dict(dict_input))
        dict_input=dict(dict_input)
    boo_is_iterable= (type(dict_input)==type(list())) or (type(dict_input)==type(tuple()))
    if boo_is_iterable:
        dict_output_lst=[]
        for value in dict_input:
            value_dict=convert_to_dict_recursive(value)
            dict_output_lst.append(value_dict)
        return dict_output_lst
    boo_is_dict= type(dict_input)==type(dict())
    if boo_is_dict:
        dict_output={}
        for key in list(dict_input.keys()):
            value=dict_input[key]
            value_=convert_to_dict_recursive(value)
            dict_output[key]=value_
        return dict_output
    boo_is_bool= (type(dict_input)==type(bool())) or (type(dict_input)==np.bool_)
    if boo_is_bool:
        dict_input=int(dict_input)
    return dict_input

###################################################
# Aliases
###################################################
def save_dict_to_json_converted(pdict,save_fn,indent=1,sort_keys=False):
    """Converts pandas data to dict data and then saves dict or list of pandas/numpy data to .json_file

    Example Usage:
save_dict_to_json_converted(pdict,save_fn)
    """
    dict_output=convert_to_dict_recursive(dict_input=pdict)
    return save_dict_to_json(dict_output,save_fn,indent=indent,sort_keys=sort_keys)

def save_to_json_converted(pdict,save_fn,indent=1,sort_keys=False):
    return save_dict_to_json_converted(pdict,save_fn,indent=indent,sort_keys=sort_keys)
