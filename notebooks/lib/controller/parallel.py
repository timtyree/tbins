#parallel.py
#Programmer: Tim Tyree
#Date: 3.20.2022
import dask.bag as db, time
#warnings.simplefilter should be able to cease incessant warnings from pandas about its depricated Int64Index class being raised by xgboost...
import warnings #intentionally redundant import 
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def eval_routine_daskbag(routine,task_lst,npartitions,printing=True,**kwargs):
    """eval_routine_daskbag returns a list of the values returned by routine, which takes a single argument, task, which is an element of the list, task_lst.
    the integer number of cores requested is npartitions.  it must be no more than os.cpu_count().
    if printing is True, then the overall run time is printed.

    Example Usage: evaluate dask.bag instance over task_lst using routine
#run daskbag
printing=True
npartitions=np.min((len(task_lst),max_num_jobs))
if printing:
    print(f"running {len(task_lst)} tasks over {npartitions} cpu cores (one task may take ~30 minutes)...")
retval=eval_routine_daskbag(routine,task_lst,npartitions,printing=printing)
if printing:
    print(f"tasks complete!")
#format results
dict_decoder_lst=[]
for rv in retval:
    dict_cm,dict_decoder=routine(task)
    dict_decoder_lst.append(dict_cm)
    """
    start = time.time()
    if npartitions>1:
        bag = db.from_sequence(task_lst, npartitions=npartitions).map(routine)
        retval = list(bag)
    else:
        retval=[]
        for task in task_lst:
            retval.append(routine(task))
    if printing:
        print(f"run time for evaluating routine was {time.time()-start:.2f} seconds, yielding {len(retval)} values returned")
    return retval
