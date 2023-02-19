#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 18:55:38 2023

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
importlib.reload(HONG)
import csv
from tqdm import tqdm
import csv
from numpy import genfromtxt
import matplotlib as mpl



prop_committed = 0.03
betas = [0.2759, 0.4]
ensemble_size = 10
run_length = 3*10**6
qs = [1, 0.5, 0]
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']



for social_structure in social_structures:
    for q in qs:
        for beta in betas:
            beta_non_committed = beta
            beta_committed = beta
            fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_q={q}_{run_length}_{ensemble_size}' 

            data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

            print('loaded data')

            A_data = data[0::3,::50]
            B_data = data[1::3,::50]
            AB_data = data[2::3,::50]

            N = A_data[0,0]+B_data[0,0]+AB_data[0,0]

            A_data = A_data/N
            B_data = B_data/N
            AB_data = AB_data/N

            A_data_av = np.median(A_data, axis=0)
            A_data_25 = np.percentile(A_data, 25, axis=0)
            A_data_75 = np.percentile(A_data, 75, axis=0)
            B_data_av = np.median(B_data, axis=0)
            B_data_25 = np.percentile(B_data, 25, axis=0)
            B_data_75 = np.percentile(B_data, 75, axis=0)
            AB_data_av = np.median(AB_data, axis=0)
            AB_data_25 = np.percentile(AB_data, 25, axis=0)
            AB_data_75 = np.percentile(AB_data, 75, axis=0)

            t = np.linspace(0, run_length, num=run_length//50+1, endpoint=True)

            plt.figure()
            plt.plot(t, A_data_av, color='tab:blue', label='A')
            plt.fill_between(t, y1=A_data_25, y2=A_data_75, color='tab:blue', alpha=0.2)
            plt.plot(t, B_data_av, color='tab:orange', label='B')
            plt.fill_between(t, y1=B_data_25, y2=B_data_75, color='tab:orange', alpha=0.2)
            plt.plot(t, AB_data_av, color='tab:green', label='A,B')
            plt.fill_between(t, y1=AB_data_25, y2=AB_data_75, color='tab:green', alpha=0.2)
            plt.xscale('log')
            plt.title(r'$\beta$' + f' = {beta_non_committed} and q={q}')
            plt.xlabel('$t$ / number of interactions')
            plt.ylabel(r'$N_{x}(t)$')
            plt.legend(title=social_structure)
            plt.savefig(f'figures/{fname}_logtime.pdf')
            plt.show()