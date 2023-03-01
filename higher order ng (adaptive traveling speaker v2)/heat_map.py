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
import seaborn as sb
import csv

def create_csv_from_list_of_lists(list_of_lists, file_name):
    with open('outputs/{file_name}', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in list_of_lists:
            writer.writerow(row)

#mpl.style.use('report_style.mplstyle')

prop_committed = np.linspace(0.01, 0.2, 20)
betas = np.linspace(0.05, 1, 20)
ensemble_size = 10
run_length = 10**6
qs = [1]
social_structures = ['InVS15']#['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']
res = 20


Bstar_arr = np.zeros((res,res))
Astar_arr = np.zeros((res,res))
for social_structure in social_structures:
    for q in qs:
        for i, p in enumerate(prop_committed):
            for j, b in enumerate(betas):
                
                fname = f'{social_structure}_{p}_{b}_{b}_q={q}_{run_length}_{ensemble_size}'
                
                data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')
    
                
    
                A_data = data[0::3,:]
                B_data = data[1::3,:]
                AB_data = data[2::3,:]
    
                N = A_data[0,0]+B_data[0,0]+AB_data[0,0]
    
                A_data = A_data/N
                B_data = B_data/N
                AB_data = AB_data/N
    
                Bstar_arr[j,i] = np.median(B_data[:, -1])
                Astar_arr[j,i] = np.median(A_data[:, -1])
                
        fname = f'heatmap_res{res}_{social_structure}_{q}_{run_length}_{ensemble_size}'
        create_csv_from_list_of_lists(Bstar_arr-Astar_arr, fname)
        # print(Bstar_arr)
        colormap = sb.color_palette(palette="RdBu", n_colors=None, desat=None, as_cmap=True)
        sb.heatmap((Bstar_arr-Astar_arr)[::-1,:], xticklabels=np.round_(prop_committed[::], decimals=3), yticklabels=np.round_(betas[::], decimals=3), cbar_kws={'label':r'$n_{B}^{\ast}-n_{A}^{\ast}$'}, cmap=colormap, center=0)
        plt.xlabel('p')
        plt.ylabel(r'$\beta$')
        plt.savefig(f'figures/{fname}.pdf')