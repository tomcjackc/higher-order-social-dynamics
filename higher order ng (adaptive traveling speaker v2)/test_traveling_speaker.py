#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:30:11 2023

@author: Marius
"""

from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from HONG import *
import HONG
import csv
from tqdm import tqdm
import csv
from numpy import genfromtxt
import matplotlib as mpl
import seaborn as sb
import pandas as pd

#mpl.style.use('report_style.mplstyle')




#%%
prop_committed = np.linspace(0.01, 0.2, 20)
betas = np.linspace(0.05, 1, 20)
ensemble_size = 10
run_length = 10**6
qs = [0, 1]
social_structures = ['InVS15']#['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

run_multiprocessing_ensamble(prop_committed, betas, ensemble_size, run_length, social_structures, qs, rule = 'Unanimous', thr = 3)
