#%%
from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
import HONG 
from HONG import *
import csv
from tqdm import tqdm
from numpy import genfromtxt
import matplotlib as mpl
import seaborn as sb
import csv


#mpl.style.use('report_style.mplstyle')

prop_committed = np.linspace(0.01, 0.2, 5)
betas = np.linspace(0.05, 1, 5)
ensemble_size = 2
run_length = 10**4
qs = [0]
social_structures = ['InVS15']#['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

create_csvs_from_outputs(prop_committed, betas, ensemble_size, run_length, social_structures, qs, sample_size=100, choose=10)

        
###Uncoment the code below to generate the actual heatmap

# colormap = sb.color_palette(palette="RdBu", n_colors=None, desat=None, as_cmap=True)
# sb.heatmap((Bstar_arr-Astar_arr)[::-1,:], xticklabels=np.round_(prop_committed[::], decimals=3), yticklabels=np.round_(betas[::], decimals=3), cbar_kws={'label':r'$n_{B}^{\ast}-n_{A}^{\ast}$'}, cmap=colormap, center=0)
# plt.xlabel('p')
# plt.ylabel(r'$\beta$')
# plt.savefig(f'figures/{fname}.pdf')
#%%
