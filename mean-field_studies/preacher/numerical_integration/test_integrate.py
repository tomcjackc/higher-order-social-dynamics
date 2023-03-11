#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:05:59 2023

@author: Marius
"""

from integrate import *




ps = np.linspace(0.01, 0.2, 5)
betas = np.linspace(0.05, 1, 5)
run_length = 10**4
qs = [0]
social_structures = ['InVS15']
import warnings
warnings.filterwarnings("ignore")

# run_multiprocessing_ensemble(ps, betas, run_length, social_structures, qs)
create_csvs_from_outputs(ps, betas, run_length, social_structures, qs)

#%%