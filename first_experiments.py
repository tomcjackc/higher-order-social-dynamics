#%%
from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from imports import *
#importlib.reload(imports)
import csv
from tqdm import tqdm
#%%


prop_committed = 0.03
beta_non_committed = 0.41
beta_committed = 0.41
ensemble_size = 50
run_length = 10**6
social_structures = [ 'InVS15', 'LyonSchool', 'SFHH', 'Thiers13']

for i in social_structures:
    run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, i)

#%%
