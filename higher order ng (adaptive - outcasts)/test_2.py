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

prop_committed = 0.08
beta_non_committed = 1
beta_committed = 1
ensemble_size = 4
run_length = 10**5
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for i in social_structures[1:2]:
    run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, i, q_non_committed=1, q_committed=1)



social_structures = {'InVS15':0, 'LyonSchool':0, 'SFHH':0, 'Thiers13':0}

for social_structure in list(social_structures.keys())[1:2]:

    output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{1}_{1}_{run_length}_{ensemble_size}'

    data = genfromtxt(f'outputs/{output_fname}.csv', delimiter=',')

    print('loaded data')

    A_data = data[0::4,:]
    B_data = data[1::4,:]
    AB_data = data[2::4,:]
    singleton_data = data[3::4,:]

    N = A_data[0,0]+B_data[0,0]+AB_data[0,0]

    A_data = A_data/N
    B_data = B_data/N
    AB_data = AB_data/N
    singleton_data = singleton_data/N

    A_data_av = np.median(A_data, axis=0)
    A_data_25 = np.percentile(A_data, 25, axis=0)
    A_data_75 = np.percentile(A_data, 75, axis=0)
    B_data_av = np.median(B_data, axis=0)
    B_data_25 = np.percentile(B_data, 25, axis=0)
    B_data_75 = np.percentile(B_data, 75, axis=0)
    AB_data_av = np.median(AB_data, axis=0)
    AB_data_25 = np.percentile(AB_data, 25, axis=0)
    AB_data_75 = np.percentile(AB_data, 75, axis=0)
    singleton_data_av = np.median(singleton_data, axis=0)
    singleton_data_25 = np.percentile(singleton_data, 25, axis=0)
    singleton_data_75 = np.percentile(singleton_data, 75, axis=0)
    print(f'Final B density = {B_data_av[-1]}')

    t = np.linspace(0, run_length, num=run_length+1, endpoint=True)

    plt.figure()
    plt.plot(t, A_data_av, color='tab:blue', label='A')
    plt.fill_between(t, y1=A_data_25, y2=A_data_75, color='tab:blue', alpha=0.2)
    plt.plot(t, B_data_av, color='tab:orange', label='B')
    plt.fill_between(t, y1=B_data_25, y2=B_data_75, color='tab:orange', alpha=0.2)
    plt.plot(t, AB_data_av, color='tab:green', label='A,B')
    plt.fill_between(t, y1=AB_data_25, y2=AB_data_75, color='tab:green', alpha=0.2)
    plt.plot(t, singleton_data_av, color='black', label='Singleton')
    plt.fill_between(t, y1=singleton_data_25, y2=singleton_data_75, color='black', alpha=0.2)
    plt.xscale('log')
    plt.title(r'$\beta$' + f' = {beta_non_committed}')
    plt.xlabel('$t$ / number of interactions')
    plt.ylabel(r'$N_{x}(t)$')
    plt.legend(title=r'$x$')
    plt.savefig(f'figures/{output_fname}_logtime.pdf')
    plt.show()