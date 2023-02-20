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
prop_committed = 0.03
betas = np.linspace(0, 1, 51)
ensemble_size = 10
run_length = 10**6
qs = [1, 0.5, 0]
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for social_structure in social_structures[3:]:
    for q in qs:
        for beta in betas:
            run_ensemble_experiment(prop_committed, beta, beta, ensemble_size, run_length, social_structure, q)
        