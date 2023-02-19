#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:15:55 2023

@author: Marius
"""

from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
import HONG
import csv
from tqdm import tqdm
from numpy import genfromtxt
import matplotlib as mpl

#mpl.style.use('report_style.mplstyle')

prop_committed = 0.03
betas = np.linspace(0, 1, 51)
ensemble_size = 10
run_length = 10**6
qs = [0,0.5, 1]
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']
for social_structure in social_structures[2:3]:
    for q in [1]:
        for beta in [0.14]:
            fname = f'{social_structure}_{prop_committed}_{beta}_{beta}_q={q}_{run_length}_{ensemble_size}'

            data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')
            
            
            if data.shape[0] != ensemble_size*3:
                print('Problem with '+fname)
                print(data.shape[0])
            else:
                print('All is good in '+fname)
                
            


