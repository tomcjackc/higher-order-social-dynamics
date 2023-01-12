#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:41:22 2022

@author: Marius
"""

from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools

import HONG as ng

import csv
from tqdm import tqdm
import csv
from numpy import genfromtxt
import matplotlib as mpl
import pandas as pd

#mpl.style.use('report_style.mplstyle')

def eq_time(A_data):
    ensemble_size = 0
    count = []
    N =  A_data.shape[1]
    for A in A_data:
        i  = A.index(0.0)
        count.append(i)
        # for i in range(N):
        #     if A[i] == 0.0:
        #         count += [i]
                
        #         break
    if len(count) > 0:
        return (np.mean(count), np.percentile(count, 25), np.percentile(count, 75))
    else:
        return (np.nan, np.nan, np.nan)


#%%

prop_committed = 0.03
betas = [0.4] #np.linspace(0, 1, 51)
ensemble_size =  50
run_length = 10**6
fs = np.linspace(0, 0.5, 51)
sample_size = 5*10**4

social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']


### TO CHANGE THE NAME FOR SAVING TO INCLUDE f!!

for social_structure in social_structures:
    for beta in betas:
        for f in fs:
            prop_committed_B = f*prop_committed
            prop_committed_C = (1-f)*prop_committed
            ng.run_ensemble_experiment(prop_committed_B,prop_committed_C , beta, beta, ensemble_size, run_length, social_structure)
            



#%%
dic_experiment = {'Astar':[],
                'Astar25': [],
                'Astar75': [],
                'Bstar': [],
                'Bstar25': [],
                'Bstar75': [],
                'Cstar':[],
                'Cstar25': [],
                'Cstar75': [],
                'beta': [],
                'f' : [],
                'equilibrium_time': [],
                'equilibrium_time25': [],
                'equilibrium_time75': [],
                'prop_committed':[]}


for social_structure in social_structures:
    for f in fs:
        for beta in betas:
            beta_non_committed = beta
            beta_committed = beta
            prop_committed_B = f*prop_committed
            prop_committed_C = (1-f)*prop_committed
            fname = f'{social_structure}_B={prop_committed_B}_C={prop_committed_C}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'
    
            data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')
    
            print(f'loaded data: {beta}')
            
            
            
            A_data = data[0::4,:]
            
            B_data = data[1::4,:]
            C_data = data[2::4,:]
            Mixed_data = data[3::4,:]
    
            N = A_data[0,0]+B_data[0,0]+C_data[0,0]+Mixed_data[0,0]
    
            A_data = A_data/N
            B_data = B_data/N
            C_data = C_data/N
            Mixed_data = Mixed_data/N
            
            ### TO CREATE A FUNCTION THAT CALCULATES THE CONVERGENCE TIME FOR A_data
            
            
            
            del_t, del_t25, del_t75 = eq_time(A_data)
            dic_experiment['equilibrium_time'] += [del_t]
            dic_experiment['equilibrium_time25'] += [del_t25]
            dic_experiment['equilibrium_time75'] += [del_t75]
            
            
            
            
            
            
    
            indices = np.random.choice(np.arange(run_length - sample_size, run_length, 1), size = 100, replace = False)
    
            A_data = A_data.T[indices]
            
            A_data = np.mean(A_data, axis = 0)
            
    
            dic_experiment['Astar'] += [np.median(A_data)]
            dic_experiment['Astar25'] += [np.percentile(A_data, 25)]
            dic_experiment['Astar75'] += [np.percentile(A_data, 75)]
    
            
    
            B_data = B_data.T[indices]
    
            B_data = np.mean(B_data, axis = 0)
    
            dic_experiment['Bstar'] += [np.median(B_data)]
            dic_experiment['Bstar25'] += [np.percentile(B_data, 25)]
            dic_experiment['Bstar75'] += [np.percentile(B_data, 75)]
    
            
    
            C_data = C_data.T[indices]
    
            C_data = np.mean(C_data, axis = 0)
    
            dic_experiment['Cstar'] += [np.median(C_data)]
            dic_experiment['Cstar25'] += [np.percentile(C_data, 25)]
            dic_experiment['Cstar75'] += [np.percentile(C_data, 75)]
            
            dic_experiment['beta'] += [beta]
            dic_experiment['f'] += [f]
            dic_experiment['prop_committed'] += [prop_committed]
    
    
    fname = f'{social_structure}_equilibrium_state_beta_p={prop_committed}_{run_length}_{ensemble_size}.csv'
    df = pd.DataFrame(dic_experiment)
    df.to_csv(f'eq_time_stats/{fname}', index =False)
        

