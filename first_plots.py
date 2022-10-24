#%%
from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from imports import *
import csv
from tqdm import tqdm
from numpy import genfromtxt



#%%

# linear time axis

prop_committed = 0.03
beta_non_committed = 0.28
beta_committed = 0.28
ensemble_size = 50
run_length = 10**6
social_structure = 'InVS15'

fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'

data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

print('loaded data')

A_data = data[0::3,:]
B_data = data[1::3,:]
AB_data = data[2::3,:]

A_data_av = np.median(A_data, axis=0)
A_data_25 = np.percentile(A_data, 25, axis=0)
A_data_75 = np.percentile(A_data, 75, axis=0)
B_data_av = np.median(B_data, axis=0)
B_data_25 = np.percentile(B_data, 25, axis=0)
B_data_75 = np.percentile(B_data, 75, axis=0)
AB_data_av = np.median(AB_data, axis=0)
AB_data_25 = np.percentile(AB_data, 25, axis=0)
AB_data_75 = np.percentile(AB_data, 75, axis=0)

t = np.linspace(0, run_length, num=run_length+1, endpoint=True)

plt.figure(1)
plt.plot(t, A_data_av, color='tab:blue', label='A')
plt.fill_between(t, y1=A_data_25, y2=A_data_75, color='tab:blue', alpha=0.2)
plt.plot(t, B_data_av, color='tab:orange', label='B')
plt.fill_between(t, y1=B_data_25, y2=B_data_75, color='tab:orange', alpha=0.2)
plt.plot(t, AB_data_av, color='tab:green', label='A,B')
plt.fill_between(t, y1=AB_data_25, y2=AB_data_75, color='tab:green', alpha=0.2)
plt.legend()
plt.xlabel(r'Time, $t$ / number of interactions')
plt.ylabel(r'$N_{x}(t)$')
plt.legend(title=r'$x$')
plt.savefig(f'figures/{fname}_lintime.pdf')
plt.show()

#%%
"""
THIS SECTION CAN REPRODUCE GIOVANI'S PLOTS FROM PAGE 4

WE SHOULD CREATE ON PAIR FOR BETA = 0.28 AND ONE PAIR FOR BETA = 0.41
AND PRESENT THOSE IN THE MEETING

"""


prop_committed = 0.03
beta_non_committed = 0.28
beta_committed = 0.28
ensemble_size = 50
run_length = 10**6

social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for social_structure in social_structures:
    fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'

    data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

    print('loaded data')

    
    B_data = data[1::3,:]
    

    
    B_data_av = np.median(B_data, axis=0)
    B_data_25 = np.percentile(B_data, 25, axis=0)
    B_data_75 = np.percentile(B_data, 75, axis=0)
    

    t = np.linspace(0, run_length, num=run_length+1, endpoint=True)

    

    plt.plot(t, B_data_av, label = social_structure)
    plt.fill_between(t, y1=B_data_25, y2=B_data_75, alpha=0.2)


    plt.xlabel('Time, $t$ / number of interactions', fontsize=16)
    plt.ylabel(r'$n_{B+B_c}(t)$', fontsize=16)

plt.legend()
plt.title
plt.savefig(f'figures/{fname}_lintime.pdf')
plt.show()


prop_committed = 0.03
beta_non_committed = 0.28
beta_committed = 0.28
ensemble_size = 50
run_length = 10**6

social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for social_structure in social_structures:
    fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'

    data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

    print('loaded data')

    
    AB_data = data[2::3,:]
    

    AB_data_av = np.median(AB_data, axis=0)
    AB_data_25 = np.percentile(AB_data, 25, axis=0)
    AB_data_75 = np.percentile(AB_data, 75, axis=0)
    

    t = np.linspace(0, run_length, num=run_length+1, endpoint=True)

    

    plt.plot(t, AB_data_av, label = social_structure)
    plt.fill_between(t, y1=AB_data_25, y2=AB_data_75, alpha=0.2)


    plt.xlabel('Time, $t$ / number of interactions', fontsize=16)
    plt.ylabel(r'$n_{AB}(t)$', fontsize=16)

plt.legend()
plt.title
plt.savefig(f'figures/{fname}_lintime.pdf')
plt.show()




#%%

# logarithmic time axis

prop_committed = 0.03
beta_non_committed = 0.28
beta_committed = 0.28
ensemble_size = 10
run_length = 500
social_structures = ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for social_structure in social_structures:
    fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'

    data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

    print('loaded data')

    A_data = data[0::3,:]
    B_data = data[1::3,:]
    AB_data = data[2::3,:]

    A_data_av = np.median(A_data, axis=0)
    A_data_25 = np.percentile(A_data, 25, axis=0)
    A_data_75 = np.percentile(A_data, 75, axis=0)
    B_data_av = np.median(B_data, axis=0)
    B_data_25 = np.percentile(B_data, 25, axis=0)
    B_data_75 = np.percentile(B_data, 75, axis=0)
    AB_data_av = np.median(AB_data, axis=0)
    AB_data_25 = np.percentile(AB_data, 25, axis=0)
    AB_data_75 = np.percentile(AB_data, 75, axis=0)

    t = np.linspace(0, run_length, num=run_length+1, endpoint=True)

    plt.figure()
    plt.plot(t, A_data_av, color='tab:blue', label='A')
    plt.fill_between(t, y1=A_data_25, y2=A_data_75, color='tab:blue', alpha=0.2)
    plt.plot(t, B_data_av, color='tab:orange', label='B')
    plt.fill_between(t, y1=B_data_25, y2=B_data_75, color='tab:orange', alpha=0.2)
    plt.plot(t, AB_data_av, color='tab:green', label='A,B')
    plt.fill_between(t, y1=AB_data_25, y2=AB_data_75, color='tab:green', alpha=0.2)
    plt.xscale('log')
    plt.xlabel('Time, $t$ / number of interactions')
    plt.ylabel(r'$N_{x}(t)$')
    plt.legend(title=r'$x$')
    plt.savefig(f'figures/{fname}_logtime.pdf')
    plt.show()



#%%