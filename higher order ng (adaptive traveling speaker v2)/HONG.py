#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:05:07 2023

@author: Marius
"""

from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random as rand
import itertools
from tqdm import tqdm
import csv
import os
import multiprocessing
from numpy import genfromtxt
import pandas as pd

#%%

# model constructor

class HigherOrderNamingGame(xgi.Hypergraph):
    def __init__(self, rule='Unanimous', incoming_data=None, **attr):
        xgi.Hypergraph.__init__(self, incoming_data, **attr)
        self.rule = rule
        
    def add_naming_game_node(self, list_nodes, vocab, committed=False, beta=1,q=0, meta=None):
        """Adds a set of identical naming game nodes, with defined vocabularies, levels of
        committment, beta, and optional metadata.

        Args:
            list_nodes (list): list of node labels to be added to the hypergraph.
            vocab (list): list representing the vocabulary shared by the added nodes.
            committed (bool, optional): the committment status of the added nodes. Defaults to False.
            beta (int, optional): the beta value (representing synergy/efficiency of communication/social influence) of the added nodes. Defaults to 1.
            meta (_type_, optional): optional metadata associated with the addded nodes (eg. whether the node is a teacher or a student in a school). Defaults to None.

        Returns:
            None
        """
        # label = self.num_nodes
        N = len(list_nodes)
        
        list_dict = [{'vocab':vocab, 'committed':committed, 'beta':beta, 'q':q}]*N
        self.add_nodes_from(zip(list_nodes, list_dict))
        return None
    
    def clear_edges(self):
        """Remove all edges from the graph without altering any nodes. This is an alteration to the XGI code and was implemented in PR #199."""
        for node in self.nodes:
            self._node[node] = set()
        self._edge.clear()
        self._edge_attr.clear()

    def interact_and_advance(self, verbose=False):
        """Function which carries out one interaction of the naming game on a given edge, and advances to the next frame.

        Args:
            edge (list): list of nodes existing on the selected edge.
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing the changes made to the counts of each vocabulary, used for counting efficiently.
        """
        
        ### Need to pick random edge inside the this func
        edges = self.edges
        edge_id = rand.choice(edges)
        edge = list(self.edges.members(edge_id))
        speaker = rand.choice(edge)
        before_dict = self.count_by_vocab_in_edge(edge)
        #print(list(self.nodes.attrs), '\n')
        
        if verbose:
            #print(f'{self.edges.members()}')
            print(f'Edge: {edge}')
            print(f'Speaker: {speaker}')
        
        broadcast = random.choice(self.get_attr(speaker, 'vocab'))

        test_stat = np.random.binomial(1, self.get_attr(speaker, 'beta'))
        test_stat_q = np.random.binomial(1, self.get_attr(speaker, 'q'))
        edge_size_change_index = False
        if self.rule == 'Unanimous':
            if all([broadcast in self.get_attr(i, 'vocab') for i in edge]):
                # if verbose:
                #     print('agreement possible')
                if test_stat:
                    # if verbose:
                    #     print('beta condition satisfied')
                    for j in edge:
                        if not self.get_attr(j, 'committed'): # sets all listener nodes to vocab=broadcast
                            xgi.classes.function.set_node_attributes(self, {j: {'vocab':[broadcast]}})
                            
                else:
                    pass
            else:
                
                for j in edge:
                    if broadcast not in self.get_attr(j, 'vocab') and not self.get_attr(j, 'committed'):
                        
                        xgi.classes.function.set_node_attributes(self, \
                        {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
                if test_stat_q:
                    #rewire
                    
                    edge_size_change_index = self.move_speaker(speaker, edge_id)
                            
                        
        d_edge_size_change = np.zeros((self.N))             
        if edge_size_change_index != False:
            d_edge_size_change[edge_size_change_index[0]-1] += -1/self.M
            d_edge_size_change[edge_size_change_index[0]-2] += 1/self.M
            d_edge_size_change[edge_size_change_index[1]-1] += -1/self.M
            d_edge_size_change[edge_size_change_index[1]] += 1/self.M
            #print(d_edge_size_change[:20])
                               
        # if self.rule == 'Union':
        #     if sum([1 for i in edge if broadcast in self.get_attr(i, 'vocab')]) > 1:
        #          if test_stat:
        #              for i in edge:
        #                  if not self.get_attr(i, 'committed'): # sets all listener nodes to vocab=broadcast
        #                      xgi.classes.function.set_node_attributes(self, {i:{'vocab':[broadcast]}})
        #              if not self.get_attr(speaker, 'committed'): # sets speaker node to vocab=broadcast
        #                  xgi.classes.function.set_node_attributes(self, {speaker:{'vocab':[broadcast]}})
        #          else:
        #              for i in edge:
        #                  if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
        #                      self.get_attr(i, 'vocab').append(broadcast)
        #     else:
        #          for i in edge:
        #              if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
        #                      xgi.classes.function.set_node_attributes(self, \
        #                     {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
        
        after_dict = self.count_by_vocab_in_edge(edge)
        diff_dict = {}
        
        for i in after_dict.keys():
            diff_dict[i] = after_dict[i]-before_dict[i]
            
        if verbose:
            
            print(f'Word broadcast: {broadcast}')
            print(before_dict)
            print(after_dict)
            print(diff_dict)
            print(list(self.nodes.attrs))
            print()
            
        return diff_dict, d_edge_size_change
    def move_speaker(self, speaker_id, edge_id):
        """
        The func removes the speaker and adds it to a new edge 
        choosen at random from the graph 
        

        Parameters
        ----------
        speaker_id : TYPE
            DESCRIPTION.
        edge_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        new_edge_id = rand.choice(self.edges)
        
        while new_edge_id in self.nodes.memberships(speaker_id):
            new_edge_id = rand.choice(self.edges)
        a =  len(list(self.edges.members(edge_id)))
        self.remove_node_from_edge(edge_id, speaker_id)
        b =  len(list(self.edges.members(new_edge_id)))
        self.add_node_to_edge(new_edge_id, speaker_id)
        #print([a, b])
        return([a, b])
        
        
        
        
        
        # while new_edge_id == edge_id:
        #     new_edge_id = rand.choice(self.edges)
        # self.remove_node_from_edge(edge_id, speaker_id)
        # self.add_node_to_edge(new_edge_id, speaker_id)
       
        
                
        
    
    def get_attr(self, node, attr):
        """Function to easily obtain a given attribute from a given node in a hypergraph.

        Args:
            node (int): Integer label of the node being interrogated.
            attr (str): Name of the required attribute.

        Returns:
            (Various types): The requested attribute of the given node.
        """
        return xgi.stats.nodestats.attrs(self, [node], attr)[node]
    
    def count_by_attr(self, attr, val, norm=False):
        if norm:
            return len(self.nodes.filterby_attr(attr, val))/self.num_nodes
        else:
            return len(self.nodes.filterby_attr(attr, val))
            
    def count_by_vocab_in_edge(self, edge):

        count_dict = {'A':0, 'B':0, 'AB':0}

        for i in edge:
            if self.get_attr(i, 'vocab') == ['A'] or self.get_attr(i, 'vocab') == ['B']:
                
                count_dict[self.get_attr(i, 'vocab')[0]] += 1
            elif self.get_attr(i, 'vocab') == ['A', 'B'] or self.get_attr(i, 'vocab') == ['B', 'A']:
                
                count_dict['AB'] += 1
        return count_dict

    def run(self, edges, runlength, verbose=False):
        """runs a complete naming game on a given set of edges

        Args:
            edges (list, ndim=3): list of edges in the hypergraph. axis=0 refers to timesteps, although we don't consider time resolved data yet, so this axis should always have len=1.
                                    Axis=1 is a list of edges, axis=2 is a list of nodes in a given edge
            runlength (int): number of interactions to compute
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing a list of length=runlength for each possible vocabulary. shows the evolution of the number of agents with a given vocabulary over time.
        """
        self.N = len(self.nodes.ids)
        self.M = len(self.edges.size.aslist())
        print(self.N)
        print(self.M)
        vocab_counts = {'A':np.zeros((runlength+1)), 'B':np.zeros((runlength+1)), 'AB':np.zeros((runlength+1)), 'edge_size_dist': np.zeros((runlength+1, self.N))}
        vocab_counts['A'][0] = self.count_by_attr('vocab', ['A'], False)
        vocab_counts['B'][0] = self.count_by_attr('vocab', ['B'], False)
        vocab_counts['AB'][0] = self.count_by_attr('vocab', ['A', 'B'], False)+self.count_by_attr('vocab', ['B', 'A'], False)
        
        vocab_counts['edge_size_dist'][0] = count_lists(self.edges.size.aslist(), self.N)
        #print(vocab_counts['edge_size_dist'])
        for i in range(runlength):
            diff_dict, d_edge_size_change = self.interact_and_advance(verbose=verbose)
            vocab_counts['A'][i+1] = vocab_counts['A'][i] + diff_dict['A']
            vocab_counts['B'][i+1] = vocab_counts['B'][i] + diff_dict['B']
            vocab_counts['AB'][i+1] = vocab_counts['AB'][i] + diff_dict['AB']
            vocab_counts['edge_size_dist'][i+1] = vocab_counts['edge_size_dist'][i] + d_edge_size_change
        vocab_counts['final_n_connected'] = xgi.algorithms.connected.number_connected_components(self)
            # if vocab_counts['AB'][i+1] == 0 and vocab_counts['A'][i+1] == 0:
            #     vocab_counts['AB'][i+2:] = np.zeros((runlength-i-1))
            #     vocab_counts['A'][i+2:] = np.zeros((runlength-i-1))
            #     vocab_counts['B'][i+2:] = np.ones((runlength-i-1))
            #     vocab_counts['time_to_consensus'] = i+1
            #     break
        #vocab_counts['time_to_consensus'] = vocab_counts.get('time_to_consensus', np.inf)
        return vocab_counts

def count_lists(lst, N):
    '''
    This function needs to be changed and made fast,
    it all deoebds on lst format

    Parameters
    ----------
    lst : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    '''

    d = np.zeros((N))
    #print(N)
    total_lists = len(lst)
    for size in lst:
            d[size-1] += 1

    d /= total_lists
    return d



def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id



def run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, social_structure, q, rule, thr):
    
    ### this line can be changed depending on which threshold we would like to use, 2 is our data, and data relating to other values come from https://github.com/iaciac/higher-order-NG
    edges, unique_id = get_edges_and_uniques(f'../data/aggr_15min_cliques_thr{thr}_{social_structure}.json')
    ###
    no_edges = len(edges[0])
    
    output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_q={q}_{run_length}_{ensemble_size}'
    
    ### This part deletes a file if it already exists
    if os.path.exists(f"outputs/{output_fname}.csv"):
        os.remove(f"outputs/{output_fname}.csv")
    if os.path.exists(f"aux_outputs/sim_edge_pdf_{output_fname}.csv"):
        os.remove(f"aux_outputs/sim_edge_pdf_{output_fname}.csv")
    if os.path.exists(f"aux_outputs/sim_edge_std_{output_fname}.csv"):
        os.remove(f"aux_outputs/sim_edge_std_{output_fname}.csv")
    if os.path.exists(f'con_com_outputs/avg_{output_fname}.csv'):
        os.remove(f'con_com_outputs/avg_{output_fname}.csv')
    if os.path.exists(f'con_com_outputs/std_{output_fname}.csv'):
        os.remove(f'con_com_outputs/std_{output_fname}.csv')
    ###
    
    edge_size = np.zeros((ensemble_size,run_length+1 ,len(unique_id)))
    component_size = np.zeros((ensemble_size))
    for k in tqdm(range(ensemble_size)):
        H = HigherOrderNamingGame(rule=rule)

        number_committed = round(len(unique_id)*prop_committed)
        rand.shuffle(unique_id)

        committed_nodes, uncommitted_nodes = np.split(np.array(unique_id), [number_committed])
        
        H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=beta_non_committed, q = q)
        
        H.add_naming_game_node(committed_nodes, ['B'], True, beta=beta_committed, q = q)
        
        H.add_edges_from(edges[0])
        
        
        
        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats = H.run(edges,run_length, False)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])
        # with open(f'con_com_outputs/{output_fname}.csv', 'a') as g:
        #     write = csv.writer(g)
        #     write.writerow([stats['final_n_connected']])
        edge_size[k] = stats['edge_size_dist']
        component_size[k] = stats['final_n_connected']
    
    df2 = pd.DataFrame( np.mean(edge_size, axis = 0).T,index = range(1,len(unique_id) +1), columns =range(run_length+1))
    df3 = pd.DataFrame( np.std(edge_size, axis = 0).T,index = range(1,len(unique_id) +1), columns =range(run_length+1))
    df2.to_csv(f'aux_outputs/sim_edge_pdf_{output_fname}.csv')
    df3.to_csv(f'aux_outputs/sim_edge_std_{output_fname}.csv')       
    
   
    
    df4 = pd.DataFrame( [np.mean(component_size)])
    df5 = pd.DataFrame( [np.std(component_size)])
    df4.to_csv(f'con_com_outputs/avg_{output_fname}.csv')
    df5.to_csv(f'con_com_outputs/std_{output_fname}.csv')       
                
        
            
def run_multiprocessing_ensamble(prop_committed, betas, ensemble_size, run_length, social_structures, qs, rule = 'Unanimous', thr = 3):

    
    args = []
    for social_structure in social_structures:
        for p in prop_committed:
            for b in betas:
                for q in qs:
                    p = round(p, 2)
                    b = round(b, 2)
                    args.append((p,b,b,ensemble_size, run_length, social_structure, q, rule, thr))
    
   
    with multiprocessing.Pool() as pool:
        # Use the pool to map the function to the arguments
        print(args)
        pool.starmap(run_ensemble_experiment, args)
        
def create_csvs_from_outputs(prop_committed, betas, ensemble_size, run_length, social_structures, qs, sample_size =5*10**4, m = 100):
    
    Bstar = np.zeros((len(betas), len(prop_committed)))
    Astar = np.zeros((len(betas), len(prop_committed)))
    Bstar_25 = np.zeros((len(betas), len(prop_committed)))
    Astar_25 = np.zeros((len(betas), len(prop_committed)))
    Bstar_75 = np.zeros((len(betas), len(prop_committed)))
    Astar_75 = np.zeros((len(betas), len(prop_committed)))
    component_size_avg = np.zeros((len(betas), len(prop_committed)))
    component_size_std = np.zeros((len(betas), len(prop_committed)))
    
    
    for social_structure in social_structures:
        for q in qs:
            for i, p in enumerate(prop_committed):
                for j, b in enumerate(betas):
                    p = round(p, 2)
                    b = round(b, 2)
                    prop_committed[i] = p
                    betas[j] = b


                    fname = f'{social_structure}_{p}_{b}_{b}_q={q}_{run_length}_{ensemble_size}'
                    
                    data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')
                    data2 = pd.read_csv(f'con_com_outputs/avg_{fname}.csv', index_col=0)
                    data3 = pd.read_csv(f'con_com_outputs/std_{fname}.csv', index_col=0)
                    
                    A_value, A_25, A_75, B_value, B_25, B_75 = steady_state_preprocessing(data, run_length, sample_size, m)
                    
                    Bstar[j,i] = B_value
                    Astar[j,i] = A_value
                    Bstar_25[j,i] = B_25
                    Astar_25[j,i] = A_25
                    Bstar_75[j,i] = B_75
                    Astar_75[j,i] = A_75
                    component_size_avg[j,i] = data2.iloc[0][0]
                    component_size_std[j,i] = data3.iloc[0][0]
                    print(p,b)

            fname = f'{len(prop_committed)}x{len(betas)}_{social_structure}_{q}_{run_length}_{ensemble_size}'
            df = pd.DataFrame(Bstar, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_B_res_{fname}.csv')
            df = pd.DataFrame(Astar, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_A_res_{fname}.csv')
            df = pd.DataFrame(Bstar_25, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_B25_res_{fname}.csv')
            df = pd.DataFrame(Astar_25, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_A25_res_{fname}.csv')
            df = pd.DataFrame(Bstar_75, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_B75_res_{fname}.csv')
            df = pd.DataFrame(Astar_75, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_A75_res_{fname}.csv')
            df = pd.DataFrame(component_size_avg, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_components_avg_{fname}.csv')
            df = pd.DataFrame(component_size_std, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_components_std_{fname}.csv')
    
def steady_state_preprocessing(data, run_length, sample_size =5*10**4, m = 100):
                A_data = data[0::3,:]
                B_data = data[1::3,:]
                AB_data = data[2::3,:]
    
                N = A_data[0,0]+B_data[0,0]+AB_data[0,0]
    
                A_data = A_data/N
                B_data = B_data/N
                AB_data = AB_data/N
                indices = np.random.choice(np.arange(run_length - sample_size, run_length, 1), size = m, replace = False)
                
                A_data = A_data.T[indices]                
                A_data = np.mean(A_data, axis = 0)
                A_value = np.median(A_data)
                A_25 = np.percentile(A_data, 25)
                A_75 = np.percentile(A_data, 75)
                
                B_data = B_data.T[indices]
                B_data = np.mean(B_data, axis = 0)
                B_value = np.median(B_data)
                B_25 = np.percentile(B_data, 25)
                B_75 = np.percentile(B_data, 75)
                
                
                return A_value, A_25, A_75, B_value, B_25, B_75
            
def delete_csvs(prop_committed, betas, ensemble_size, run_length, social_structures, qs):
    for social_structure in social_structures:
        for q in qs:
            for i, p in enumerate(prop_committed):
                for j, b in enumerate(betas):
                    p = round(p, 2)
                    b = round(b, 2)

                    fname = f'{social_structure}_{p}_{b}_{b}_q={q}_{run_length}_{ensemble_size}'
                    if os.path.exists(f"outputs/{fname}.csv"):
                            os.remove(f"outputs/{fname}.csv")

if __name__ == '__main__':
    betas = [0.16, 0.28, 0.36, 0.4, 0.76]
    ps = [0.03]
    qs = [1]
    social_structures = ['InVS15']
    run_length = 10**5
    ensamble_size = 10
    import warnings
    warnings.filterwarnings("ignore")
    
    run_multiprocessing_ensamble(ps, betas, ensamble_size, run_length, social_structures, qs)
    create_csvs_from_outputs(ps, betas, ensamble_size, run_length,social_structures, qs, sample_size=100, m=20)
    # beta = 0.36
    # p = 0.03
    # q = 1
    # social_structure = 'InVS15'
    # ensamble_size = 10
    # run_length = 10**5
    # run_ensemble_experiment(p , beta, beta, ensamble_size, run_length, social_structure, q, rule = 'Unanimous', thr =3 )




# %%
