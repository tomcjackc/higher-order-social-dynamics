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

mpl.style.use('report_style.mplstyle')

prop_committed = 0.03
betas = [0.1, 0.2, 0.3]#np.linspace(0, 1, 51)
ensemble_size = 10
run_length = 10**6
qs = [0,0.5, 1]
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']
for social_structure in social_structures:
    for q in qs:
    ###    Needs to be changed to take 100 sampe avg from the last 50000 points, similiar to the paper!
        Astar_list = []
        Astar25_list = []
        Astar75_list = []
        Bstar_list = []
        Bstar25_list = []
        Bstar75_list = []
        for beta in betas:
            fname = f'{social_structure}_{prop_committed}_{beta}_{beta}_q={q}_{run_length}_{ensemble_size}'

            data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

            print(f'loaded data: {beta}')

            A_data = data[0::3,:]
            B_data = data[1::3,:]
            AB_data = data[2::3,:]

            N = A_data[0,0]+B_data[0,0]+AB_data[0,0]

            A_data = A_data/N
            B_data = B_data/N
            AB_data = AB_data/N

            Astar_list.append(np.median(A_data[:, -1]))
            Astar25_list.append(np.percentile(A_data[:, -1], 25))
            Astar75_list.append(np.percentile(A_data[:, -1], 75))

            Bstar_list.append(np.median(B_data[:, -1]))
            Bstar25_list.append(np.percentile(B_data[:, -1], 25))
            Bstar75_list.append(np.percentile(B_data[:, -1], 75))

        #plt.figure(1) if activated 
        plt.plot(betas, Astar_list, color='k', linestyle='--', label='A')
        plt.fill_between(betas, Astar25_list, Astar75_list, color='k', alpha=0.2)

        plt.plot(betas, Bstar_list, color='tab:blue', label='B')
        plt.fill_between(betas, Bstar25_list, Bstar75_list, color='tab:blue', alpha=0.2)

        plt.legend(title=r'$x$')
        plt.title(f'{social_structure}_q={q}')
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$N^{\ast}_{x}(\beta)$')
        plt.savefig(f'figures/fig3b_{social_structure}_{prop_committed}_q={q}_{run_length}_{ensemble_size}.pdf')
        plt.show()

