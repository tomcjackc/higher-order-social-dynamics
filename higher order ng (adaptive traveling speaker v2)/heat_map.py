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

prop_committed = [0.03]#np.linspace(0.01, 0.2, 20)
betas = np.linspace(0, 1, 51)
ensemble_size = 10
run_length = 10**6
qs = [1]
social_structures = ['InVS15']#['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

create_csvs_from_outputs(prop_committed, betas, ensemble_size, run_length, social_structures, qs)

        
###Uncoment the code below to generate the actual heatmap

# colormap = sb.color_palette(palette="RdBu", n_colors=None, desat=None, as_cmap=True)
# sb.heatmap((Bstar_arr-Astar_arr)[::-1,:], xticklabels=np.round_(prop_committed[::], decimals=3), yticklabels=np.round_(betas[::], decimals=3), cbar_kws={'label':r'$n_{B}^{\ast}-n_{A}^{\ast}$'}, cmap=colormap, center=0)
# plt.xlabel('p')
# plt.ylabel(r'$\beta$')
# plt.savefig(f'figures/{fname}.pdf')