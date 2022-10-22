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

prop_committed = 0.05
beta_non_committed = 0.2
beta_committed = 0.2
ensemble_size = 50
run_length = 5000
social_structure = 'InVS15'

fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'

data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')

A_data = data[0::3,:]
B_data = data[1::3,:]
AB_data = data[2::3,:]

A_data_av = np.mean(A_data, axis=0)
B_data_av = np.mean(B_data, axis=0)
AB_data_av = np.mean(AB_data, axis=0)

plt.figure(1)
plt.plot(A_data_av, color='tab:blue', label='A')
plt.plot(B_data_av, color='tab:orange', label='B')
plt.plot(AB_data_av, color='tab:green', label='A,B')
plt.legend()
plt.show()

#%%