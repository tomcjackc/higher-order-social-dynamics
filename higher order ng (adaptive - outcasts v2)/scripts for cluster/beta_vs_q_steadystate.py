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
import seaborn as sb

mpl.style.use('report_style.mplstyle')

########

res = 5
prop = 0.03
betas = np.linspace(0, 1, res)
qs = np.linspace(0, 1, res)

ensemble_size = 2
run_length = 10**6
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

########

for i in social_structures:
    for q in qs:
        for beta in betas:
            print((beta, q))
            run_ensemble_experiment(prop, beta, beta, ensemble_size, run_length, i, q_non_committed=q, q_committed=q)

########

Bstar_arr = np.zeros((res,res))
Astar_arr = np.zeros((res,res))

for i in social_structures:
    for j,q in enumerate(qs[::]):
        for k, beta in enumerate(betas[::-1]):
            # print((beta, q))
            fname = f'{i}_{prop}_{beta}_{beta}_{q}_{q}_{run_length}_{ensemble_size}'

            data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')


            A_data = data[0::3,:]
            B_data = data[1::3,:]
            AB_data = data[2::3,:]

            N = A_data[0,0]+B_data[0,0]+AB_data[0,0]

            B_data = B_data/N
            A_data = A_data/N

            Bstar_arr[k,j] = np.median(B_data[:, -1])
            Astar_arr[k,j] = np.median(A_data[:, -1])
            # print(np.median(B_data[:, -1])-np.median(A_data[:, -1]))
    
    fname = f'heatmap_res{res}_{i}_p={prop}_{run_length}_{ensemble_size}'
    sb.heatmap((Bstar_arr-Astar_arr)[::-1,:], xticklabels=np.round_(props[::], decimals=3), yticklabels=np.round_(betas[::], decimals=3), cbar_kws={'label':r'$n_{B}^{\ast}-n_{A}^{\ast}$'}, cmap="RdBu")
    plt.xlabel('p')
    plt.ylabel(r'$\beta$')
    plt.savefig(f'figures/{fname}.pdf')