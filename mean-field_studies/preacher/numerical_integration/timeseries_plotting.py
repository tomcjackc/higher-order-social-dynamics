#%%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import binom
import itertools
import multiprocessing
import csv
from numpy import genfromtxt

dist = 'LyonSchool'
p = 0.01
beta = 0.01
q = 0
t_max = 10**4
output_fname = f'{dist}_{p}_{beta}_{beta}_q={q}_{t_max}'

data = genfromtxt(f'outputs/{output_fname}.csv', delimiter=',')

print('loaded data')

A_data = data[0,:]
B_data = data[1,:]
AB_data = data[2,:]

t = np.linspace(0, t_max, num=t_max, endpoint=True)[:]

plt.figure()
plt.plot(t, A_data, color='tab:blue', label='A')
plt.plot(t, B_data, color='tab:orange', label='B')
plt.plot(t, AB_data, color='tab:green', label='A,B')
plt.xscale('log')
plt.title(r'$\beta$' + f' = {beta} and q={q}')
plt.xlabel('$t$ / number of interactions')
plt.ylabel(r'$N_{x}(t)$')
plt.legend(title=dist)
plt.xlim((0, t_max))
plt.savefig(f'figures/{output_fname}_logtime.pdf')
plt.show()
#%%