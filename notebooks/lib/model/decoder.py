# decoder.py
#Programmer: Tim Tyree
#Date: 9.3.2022
import pandas as pd, numpy as np, os
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from ..measures.auc import comp_auc_simple
from ..utils.utils import print_dict
from ..measures.point_process_measures import extract_simple_firing_rates

def fit_decoder(xtrain, ytrain, xtest=None, ytest=None, param_dict=None, verbose=0, n_estimators=21,use_label_encoder=False, **kwargs):
    '''fit_decoder trains a population level decoder to learn to predict ytrain using xtrain as its argument.
    param_dict is passed to the constructor of an XGBClassifier instance key word arguments along
    with n_estimators,use_label_encoder as keyword arguments.
    if xtest=None, ytest=None are not None, then they are used only for evaluation and not for training.

    Example Usage:
    clf=fit_decoder(xtrain, ytrain, param_dict=kwargs_decoder, printing=printing)
    '''
    if param_dict is None:
        param_dict=dict()
    #define the model
    clf = XGBClassifier(n_estimators=n_estimators,use_label_encoder=use_label_encoder,**param_dict)
    eval_set=[(xtrain, ytrain), (xtest, ytest)]
    if (xtest is None) or (ytest is None):
        eval_set=[(xtrain, ytrain)]
    #fit the model
    clf.fit(xtrain, ytrain,
            eval_set=eval_set,
            verbose=verbose)
    return clf

def test_decoder_predictions(probas,ytest,
            num_features=None,
            base_margin=0.5,
            normalize_cm=None,
                 printing=True,**kwargs):
    """
    Example Usage:
probas = clf.predict_proba(xtest)[:,1]
num_features = xtest.shape[1]
dict_cm,dict_decoder = test_decoder_predictions(probas,ytest,num_features=num_features)
    """
    #make predictions from the predictor values
    ypred = (probas>base_margin).astype(int)

    #compute auc
    auc=comp_auc_simple(label_values=ytest, predictor_values=probas)

    #compute confusion matrix with the base_margin as threshold
    cm = np.array(confusion_matrix(ytest,ypred, normalize=normalize_cm))
    FP=cm[1,0] #Type I error
    TP=cm[1,1]
    TN=cm[0,0]
    FN=cm[0,1] #Type II error

    #compute measures of prediction
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    negative_predictive_value=TN/(TN+FN)
    if TP+FN>0:
        sensitivity=TP/(TP+FN)
    else:
        sensitivity=np.nan
    if TN+FP>0:
        specificity=TN/(TN+FP)
    else:
        specificity=np.nan

    #format measures of binary classification into a dictionary instance
    dict_cm={
        'accuracy':accuracy,
        'sensitivity':sensitivity,
        'specificity':specificity,
        'precision':precision,
        'negative_predictive_value':negative_predictive_value,
        'auc':auc,
        'num_true_testing':int(np.sum(ytest==1)),
        'num_false_testing':int(np.sum(ytest==0)),
        'num_features':num_features,
    }

    #record results for plotting
    dict_decoder={
        'ypred':ypred,
        'ytest':ytest,
        'probas':probas
    }

    if printing:
        #print the bluf
        print(f"classifier model: ensemble of gradient boosted decision trees")
        print_dict(dict_cm)
        print(f"num_features={num_features}")
        #print(f"num_samples_training={ytrain.shape[0]}")
        print(f"num_samples_testing={ytest.shape[0]}")
        print(f"num_true_testing={sum(np.isclose(ytest,1.))}")
        print(f"num_false_testing={sum(np.isclose(ytest,0.))}")
        print( '************************************************************************')
        print(f"* the accuracy of the population level decoder: {dict_cm['accuracy']:.4%} (AUC={auc:.4f})")
        print( '************************************************************************')

    return dict_cm,dict_decoder


def parse_decoder_kwargs(dict_decoder_hyperparameters,task_index,gpu_available=False,scale_pos_weight=None,**kwargs):
    """parse_decoder_kwargs returns a dictionary of keyword arguements to the decoder.
    the value for scale_pos_weight in dict_decoder_hyperparameters is overwritten if scale_pos_weight is not None.
    scale_pos_weight = sum(booF)/sum(booT) is always reasonable a priori guess.

    Example Usage:
scale_pos_weight = sum(booF)/sum(booT)
gpu_available=gpu_available
kwargs_decoder = parse_decoder_kwargs(dict_decoder_hyperparameters,task_index,gpu_available=gpu_available,scale_pos_weight=scale_pos_weight)
    """
    params_text = dict_decoder_hyperparameters['params_text']['MvMMIndivConcept']
    objective_lst = params_text['objective']
    kwargs_decoder = dict(objective = objective_lst[task_index])

    if scale_pos_weight is None:
        #choose relative weight of positive trials during training
        scale_pos_weight = eval(params_text['scale_pos_weight'])[task_index]

    #parse kwargs from Table 1
    kwargs_decoder['scale_pos_weight'] = scale_pos_weight
    params_table = dict_decoder_hyperparameters['params_table']['MvMMIndivConcept']
    for key in params_table:
        val=params_table[key]
        if type(val) is not type(list()):
            val = eval(val)
        kwargs_decoder[key] = val[task_index]

    if not gpu_available:
        kwargs_decoder['tree_method'] = 'hist'
        kwargs_decoder['n_jobs'] = np.max((1,os.cpu_count()-1))
    return kwargs_decoder

def train_test_decoder(xtrain,ytrain,xtest,ytest,kwargs_decoder,
        n_epochs=60,
        verbose=0,return_clf=False,printing=False,using_gpu=False,iteration_range=None,**kwargs):
    """train_test_decoder generates predictions for a population-level decoder, clf.
    if using_gpu is False (default), then iteration_range is not used.
    otherwise, iteration_range = (1,n_epochs) and it is passed to clf.predict_proba.

    Example Usage:
dict_predictions = train_test_decoder(xtrain,ytrain,xtest,ytest,n_epochs=60,
        verbose=1,return_clf=False,printing=True)
dict_cm = dict_predictions['dict_cm']
dict_decoder = dict_predictions['dict_decoder']
print_dict(dict_cm)
    """
    #train the decoder
    clf=fit_decoder(xtrain, ytrain, xtest, ytest, param_dict=kwargs_decoder, verbose=verbose, n_estimators=n_epochs)

    #iteration_range is not currently supported without gpu in use...
    if using_gpu:
        if iteration_range is None:
            iteration_range = (1,n_epochs)
        probas = clf.predict_proba(xtest,iteration_range=iteration_range)[:,1]
    else:
        probas = clf.predict_proba(xtest)[:,1]

    #test the decoder
    num_features = xtest.shape[1]
    dict_cm,dict_decoder = test_decoder_predictions(probas,ytest,num_features=num_features,printing=printing)
    #format results
    dict_predictions=dict(dict_cm=dict_cm,dict_decoder=dict_decoder)
    if return_clf:
        dict_predictions['clf']=clf
    return dict_predictions


############################################################################
# main decoder routine for multiconcept multimodal social recognition tasks
############################################################################
def routine_generate_decoder_predictions_test(dict_tbins_lst,spike_time_array,data,d_labels,
                p_significant=0.05,
                scale_pos_weight=None, #100. #5
                learning_rate=None, #0.2
                n_epochs=200,#60
                gpu_available=False,
                return_clf=False,
                filter_unreasonable_neurons=True,
                test_selected_only=True, #
                printing=True,verbose=0,**kwargs):
    """routine_generate_decoder_predictions_test computes predictions for the population-level decoder.
    if test_selected_only is false, test with all trials that were not involved in training.
    if test_selected_only is false, then only the test trials indicated by the cv_fold in d_labels is used.
    routine_generate_decoder_predictions_test has a typical run time ~1 second per member of dict_tbins_lst.

    Example Usage:
dict_results = routine_generate_decoder_predictions_test(dict_tbins_lst,spike_time_array,data,d_labels,
                p_significant=0.05,
                scale_pos_weight=None,
                learning_rate=None,
                n_epochs=200,
                gpu_available=False,
                return_clf=False,
                test_selected_only=True,
                printing=True,verbose=0)
df_results = dict_results['df_results']
df_results.head()
    """
    #parse the time bins into tasks
    task_str_lst=[]
    df_tbins_pred_lst=[]
    df_cm_lst=[]
    dict_decoder_lst=[]
    for index,dict_tbins in enumerate(dict_tbins_lst):
        df_tbins_refined = pd.DataFrame(dict_tbins['df_tbins_refined'])
        df_tbins_pred = df_tbins_refined[df_tbins_refined['p']<p_significant].copy()
        if filter_unreasonable_neurons:
            nid_values_reasonable = dict_tbins['task']['nid_values_reasonable']
            nid_lst=df_tbins_pred['nid'].values
            nid_values_reasonable = sorted(list(dict_tbins['task']['nid_values_reasonable']))
            nid_lst_reasonable = sorted(np.intersect1d(nid_values_reasonable,nid_lst))

        task = dict_tbins['task']
        task_str = task['task_str']
        #caste python primitives in task to numpy arrays
        key_lst=['booT','booF','booT_test','booF_test','nid_values_reasonable']
        for key in key_lst:
            task[key] = np.array(task[key])

        auc_overall_values = np.max((df_tbins_pred['auc_overall'].values,1-df_tbins_pred['auc_overall'].values),axis=0)
        # if printing:
        #     print(f"predictive population for {task_str}: mean overall auc =\t{np.mean(auc_overall_values):.4f} +/- {np.std(auc_overall_values):.4f} (N_pred_tbins={auc_overall_values.shape[0]})")
        #extract predictive firing rates
        X,tbin_indices=extract_simple_firing_rates(spike_time_array, df_tbins_pred)
        #test only nid_values_reasonable are considered
        nid_values_reasonable=task['nid_values_reasonable']
        nid_values_pred = np.unique(tbin_indices[:,0]).astype(int)
        num_pred_neurons            = nid_values_pred.shape[0]
        num_pred_neurons_reasonable = np.intersect1d(nid_values_pred,nid_values_reasonable).shape[0]
        assert num_pred_neurons == num_pred_neurons_reasonable
        #print(f"\n{num_pred_neurons=}")
        #print(f"{nid_values_pred=}")

        booT=task['booT']
        booF=task['booF']
        booT_test=task['booT_test']&~task['booT']
        booF_test=task['booF_test']&~task['booF']

        #attempt to deduce the cv iteration
        boo_selected =d_labels['cv_fold_voice_only']>-1
        boo_selected|=d_labels['cv_fold_face_only']>-1
        boo_selected|=d_labels['cv_fold_match']>-1
        boo_selected|=d_labels['cv_fold_mismatch']>-1
        if test_selected_only:
            booT_test &=boo_selected
            booF_test &=boo_selected
        cv_iter = d_labels[booT_test|booF_test]['cv_fold_voice_only'].max()
        cv_iter = np.max((cv_iter, d_labels[booT_test]['cv_fold_match'].max()))

        num_trials = X.shape[0]
        #test shapes all agree
        assert num_trials == booT.shape[0]
        assert num_trials == booF.shape[0]
        assert num_trials == booT_test.shape[0]
        assert num_trials == booF_test.shape[0]
        y=np.empty(num_trials)*np.nan
        y[booT]=1.
        y[booF]=0.
        y[booT_test]=1.
        y[booF_test]=0.
        y=y.astype(int)

        #test that the sum of all trials involved is no more than the total number of trials
        assert not y.shape[0] < sum(booT) + sum(booF) + sum(booT_test) + sum(booF_test)

        boo_train = booT|booF
        boo_test  = booT_test|booF_test

        #format as labeled features as the canonical machine learning kwargs
        xtrain = X[boo_train]
        ytrain = y[boo_train]
        xtest = X[boo_test]
        ytest = y[boo_test]

        #determine decoder params
        dict_decoder_hyperparameters = dict(data['dict_decoder_hyperparameters'])
        if task_str=='MvMM':
            task_index=0
        else:
            task_index=1

        kwargs_decoder = parse_decoder_kwargs(dict_decoder_hyperparameters,task_index,gpu_available=gpu_available,scale_pos_weight=scale_pos_weight)

        #NOTE: 'n_job' was replaced from .yaml src as n_jobs. this shouldn't be needed anymore unless i goofed.
        # try: #n_job need to be removed for decoder construction to be interpreted.
        #     kwargs_decoder.pop('n_job')
        # except KeyError as e:
        #     pass
        if learning_rate is not None:
            kwargs_decoder['learning_rate']=learning_rate

        #generate predictions
        dict_predictions = train_test_decoder(xtrain,ytrain,xtest,ytest,kwargs_decoder,
                                              n_epochs=n_epochs,verbose=verbose,
                                              return_clf=return_clf,
                                              printing=False) # don't make printing=True
        dict_cm = dict_predictions['dict_cm']
        dict_decoder = dict_predictions['dict_decoder']
        dict_decoder['task_str'] = task_str
        dict_decoder['cv_iter']=cv_iter
        auc=dict_cm['auc']
        if printing:
            print(f"predictive population for {task_str}: mean overall auc =\t{np.mean(auc_overall_values):.4f} +/- {np.std(auc_overall_values):.4f} (N_pred_tbins={auc_overall_values.shape[0]}) ==> AUC={auc:.4f}")

        #record
        task_str_lst.append(task_str)
        df_cm=pd.DataFrame([dict_cm])
        df_cm['task_str'] = task_str
        df_cm['cv_iter']=cv_iter
        df_cm['lst_index'] = index
        df_cm_lst.append(df_cm)
        dict_decoder_lst.append(dict_decoder)
        df_tbins_pred_lst.append(df_tbins_pred)

    #format results
    df_results = pd.concat(df_cm_lst)
    dict_results = dict(
        df_results=df_results,
                        task_str_lst=task_str_lst,
                        dict_decoder_lst=dict_decoder_lst,
                        df_tbins_pred_lst=df_tbins_pred_lst)
    return dict_results
