# get_init.py
#Programmer: Tim Tyree
#Date: 9.8.2022

#pandas really wants you to know that they don't use Int64Index anymore when xgboost is loaded...
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# automate the boring stuff
import time, os, sys, re
import dask.bag as db
from inspect import getsource
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
    nb_dir = os.getcwd()

import numpy as np, pandas as pd, matplotlib.pyplot as plt, sys, os
import matplotlib.ticker as mtick
import matplotlib as mpl
import seaborn as sns
import time

from xgboost import XGBClassifier

# import lib
from . import *
