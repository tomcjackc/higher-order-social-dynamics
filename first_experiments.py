#%%
from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from imports import *
importlib.reload(imports)
import csv
from tqdm import tqdm
#%%
def run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, social_structure):
    edges, unique_id = get_edges_and_uniques(f'data/aggr_15min_cliques_thr2_{social_structure}.json')

    for j in tqdm(range(ensemble_size)):
        H = Hypergraph()

        temp_unique_id = unique_id.copy()

        number_committed = int(len(temp_unique_id)*prop_committed)
        committed_nodes = []
        for i in range(number_committed):
            rand_index = random.randint(0, len(temp_unique_id)-1)
            committed_nodes.append(temp_unique_id.pop(rand_index))


        for i in temp_unique_id:
            H.add_naming_game_node(i, ['A'], False, beta=beta_non_committed)
        for i in committed_nodes:
            H.add_naming_game_node(i, ['B'], True, beta=beta_committed)


        output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'
        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats = run_naming_game(H, edges, run_length, False)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])

prop_committed = 0.03
beta_non_committed = 0.28
beta_committed = 0.28
ensemble_size = 50
run_length = 10**6
social_structures = ['InVS13', 'InVS15', 'LH10', 'LyonSchool', 'SFHH', 'Thiers13']

for i in social_structures[:-1]:
    run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, i)

#%%
