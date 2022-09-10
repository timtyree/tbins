#auc.py
#Programmer: Tim Tyree
#Date: 7.11.2022
import numpy as np
from sklearn import metrics

def comp_auc_simple(label_values, predictor_values):
    """computes the area under the roc curve (auc) resulting from a sweeping threshold for binary classification.
    label_values is a 1d numpy.array of 0 or 1 ground truth labels.
    predictor_values is a set of values where values larger than larger threshold indicate a prediction of 1.

    Example Usage:
auc=comp_auc_simple(label_values, predictor_values)
    """
    fpr, tpr, thresholds = metrics.roc_curve(label_values, predictor_values)
    auc = metrics.auc(fpr, tpr)
    return auc

def comp_roc_simple(label_values, predictor_values):
    """computes the area under the roc curve (auc) resulting from a sweeping threshold for binary classification.
    label_values is a 1d numpy.array of 0 or 1 ground truth labels.
    predictor_values is a set of values where values larger than larger threshold indicate a prediction of 1.

    Example Usage:
dict_roc=comp_roc_simple(label_values, predictor_values)
tpr = dict_roc['tpr']
fpr = dict_roc['fpr']
thresholds = dict_roc['thresholds']
auc = dict_roc['auc']
print(f"overall auc: {auc:.4f}")
    """
    fpr, tpr, thresholds = metrics.roc_curve(label_values, predictor_values)
    auc = metrics.auc(fpr, tpr)
    dict_roc=dict(fpr=fpr,tpr=tpr,thresholds=thresholds,auc=auc)
    return dict_roc

def norm_predictor_quadratic(predictor_values,ideal_threshold):
    """norm_predictor_quadratic scales the magnitude of predictors
    to the same ideal_threshold=0.5.  this is useful for summarizing
    many comparable predictors of varying loudness.

    norm_predictor_quadratic returns the evaluate of the unique polynomial solution
    that mapps 0 to 0, max to 1, and ideal_threshold to 0.5.
    if ideal_threshold=0.5, then this method is equivalent to
    linearly rescaling the predictors to a maximum value of unity.

    Example Usage: test has no effect on apparent auc
fpr, tpr, thresholds = metrics.roc_curve(label_values, predictor_values)
auc = metrics.auc(fpr, tpr)
index_ideal_thresh=np.argmax(tpr*(1-fpr))
ideal_threshold = thresholds[index_ideal_thresh]
predictor_value_normed=norm_predictor_quadratic(predictor_values,ideal_threshold)
fpr, tpr, thresholds = metrics.roc_curve(label_values, predictor_value_normed)
auc_normed = metrics.auc(fpr, tpr)
assert auc_normed==auc
    """
    pat=np.array(predictor_values).flatten()
    a=ideal_threshold
    b=np.max(pat)
    assert (a<b) #where equality produces divide by zero error
    c=a*a/b
    m=(0.5-c/b)/(a-c)
    w=(1.-m*b)/b/b
    #constrain quadratic map to be strictly monotonically increasing on the interval [0,b)
    if m+2.*w*b<0:
        w=-0.5*m/b
        #print(f"{w=}") #uncomment to print the weight used by method
    y=m*pat + w*pat*pat
    return y
