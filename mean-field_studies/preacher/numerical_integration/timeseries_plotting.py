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

# betas = np.linspace(0, 1, num=4)
# ps = np.linspace(0, 0.2, num=4)
# qs = [0]
# social_structures = ['LyonSchool']
# run_length = 10**3
# import warnings
# warnings.filterwarnings("ignore")

# run_multiprocessing_ensemble(ps, betas, run_length, social_structures, qs)
# create_csvs_from_outputs(ps, betas, run_length, social_structures, qs)

#%%
# plt.figure(1)
# plt.title(f'N={sys.N}, beta={sys.beta}, p={sys.f_Bcom_init},')
# plt.plot(sys.scipy_f_A, label='f_A')
# plt.plot(sys.scipy_f_B+sys.scipy_f_Bcom, label='f_B')
# plt.plot(sys.scipy_f_AB, label='f_AB')
# # plt.plot(sys.scipy_f_Bcom, label='f_Bcom')
# plt.xscale('log')
# plt.legend(title = sys.dist)

# plt.figure(2)
# #plt.title(f'N={sys.N}, beta={sys.beta}, f_A_init={sys.f_A_init}, f_B_init={sys.f_B_init}, f_Bcom_init={sys.f_Bcom_init}, gamma={sys.gamma}, t_max={sys.t_max}')
# plt.plot(sys.scipy_M, label='Magnetisation')
# plt.xscale('log')
# plt.ylim((-1,1))
# plt.legend()
# plt.show()
# #%%
# for i in range(0,10**5,10**3):
#     pi_n = sys.res[i, 2:]
#     plt.plot(pi_n, label= f't={i}')
#     plt.ylim((0,1))
#     plt.legend()
#     plt.show()
# #%%


# plt.figure(2)
# #plt.title(f'N={sys.N}, beta={sys.beta}, f_A_init={sys.f_A_init}, f_B_init={sys.f_B_init}, f_Bcom_init={sys.f_Bcom_init}, gamma={sys.gamma}, t_max={sys.t_max}')
# plt.plot(sys.scipy_M, label='Magnetisation')
# plt.xscale('log')
# plt.ylim((-1,1))
# plt.legend()
# plt.show()
# #%%
# for i in range(0,300,10):
#     pi_n = sys.res[i, 2:]
#     #print(pi_n)
#     plt.plot(pi_n, label= f't={i}')
#     plt.ylim((0,1))
#     plt.legend()
#     plt.show()
#%%

# plt.plot(np.array([sum(sys.res[j, 2:]) for j in range(100000)]))

# plt.xscale('log')
