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

for social_structure in social_structures:
    for q in qs:
        for beta in betas:
            run_ensemble_experiment(prop_committed, beta, beta, ensemble_size, run_length, social_structure, q)
        
    
#%%

q = 1
### set params and load data
for social_structure in social_structures:
    output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_q={q}_{run_length}_{ensemble_size}' 
    data = pd.read_csv(f'aux_outputs/{output_fname}.csv', header=None)
    ###
    
    ### split data according to how it is written into the file
    initial_deg = data.iloc[0::4].to_numpy(dtype=int)
    # rewire_freq = data.iloc[1::4].to_numpy(dtype=np.int64)[:,0]
    final_deg = data.iloc[2::4].to_numpy(dtype=int)
    final_vocab = data.iloc[3::4].values.tolist()
    ###
    
    
    ### plot the initial and final degree distributions in the same bins
    plt.figure(1)
    same_bin_edges = np.linspace(0, 40, num=20, endpoint=True)
    print(same_bin_edges)
    
    #initial_bin_edges = np.histogram_bin_edges(initial_deg, bins=20)
    initial_deg_binned_freq = plt.hist(initial_deg[0], bins=same_bin_edges, label='Initial')
    
    #final_bin_edges = np.histogram_bin_edges(final_deg, bins=20)
    final_deg_binned_freq = plt.hist(final_deg[0], bins=same_bin_edges, label='Final')
    plt.legend()
    plt.xlim((0, 40))
    plt.title(social_structure)
    ###
