#%%
#hello

from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random as rand
import itertools
from tqdm import tqdm
import csv

#%%


class Hypergraph(xgi.Hypergraph):
    def add_naming_game_node(self, list_nodes, vocab, committed=False, beta=1, meta=None):
        """Adds a specified selection of nodes to the hypergraph.
        The vocabularies of the nodes can be specified.

        Args:
            uncom_recipe (dict): Specification of uncommitted nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}
            com_recipe (dict): Specification of committed nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}

        Returns:
            None
        """
        # label = self.num_nodes
        N = len(list_nodes)
        
        list_dict = [{'vocab':vocab, 'committed':committed, 'beta':beta}]*N
        self.add_nodes_from(zip(list_nodes, list_dict))
        return None
    
    def clear_edges(self):
        """Remove all edges from the graph without altering any nodes."""
        for node in self.nodes:
            self._node[node] = set()
        self._edge.clear()
        self._edge_attr.clear()

    def interact_and_advance(self, edge, rule='Unanimous', verbose=False):
        """Function which carries out one interaction in the higher-order naming game on a hypergraph and advances to the next frame.

        Args:
            edges (list): The full, time-resolved list of edges for the hypergraph. Shape should be (number of timesteps, number of edges in each timestep, number of nodes in each edge), expected to be ragged list of lists.
            frame_num (int): The index that determines which timestep is being considered.
            rule (str, optional): A description of the rule used to determine whether local consensus is possible in a given interaction. Defaults to 'Unanimous'.
            show (bool, optional): Whether or not the specifics of each interaction is shown. Useful for debugging. Defaults to False.
        
        Returns:
            None
        """
        
        
        edge = list(edge)
        speaker = rand.choice(edge)
        before_dict = self.count_by_vocab_in_edge(edge)
        #print(list(self.nodes.attrs), '\n')
        
        if verbose:
            #print(f'{self.edges.members()}')
            print(f'Edge: {edge}')
            print(f'Speaker: {speaker}')
        
        broadcast = random.choice(self.get_attr(speaker, 'vocab'))

        test_stat = np.random.binomial(1, self.get_attr(speaker, 'beta'))

        if rule == 'Unanimous':
            if all([broadcast in self.get_attr(i, 'vocab') for i in edge]):
                if test_stat:
                    #print('AGREEMENT')
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
        
                        
        if rule == 'Union':
            if sum([1 for i in edge if broadcast in self.get_attr(i, 'vocab')]) > 1:
                 if test_stat:
                     for i in edge:
                         if not self.get_attr(i, 'committed'): # sets all listener nodes to vocab=broadcast
                             xgi.classes.function.set_node_attributes(self, {i:{'vocab':[broadcast]}})
                     if not self.get_attr(speaker, 'committed'): # sets speaker node to vocab=broadcast
                         xgi.classes.function.set_node_attributes(self, {speaker:{'vocab':[broadcast]}})
                 else:
                     for i in edge:
                         if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
                             self.get_attr(i, 'vocab').append(broadcast)
            else:
                 for i in edge:
                     if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
                             xgi.classes.function.set_node_attributes(self, \
                            {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
        
        after_dict = self.count_by_vocab_in_edge(edge)
        diff_dict = {}
        
        for i in after_dict.keys():
            diff_dict[i] = after_dict[i]-before_dict[i]
            
        if verbose:
            
            print(f'Word broadcast: {broadcast}')
            print(before_dict)
            print(after_dict)
            print(diff_dict)
            
        
        
        
        return diff_dict

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




def run_naming_game(H, edges, runlength, verbose=False):
    '''
    Consider using only lists instead of Vocab Counts

    Parameters
    ----------
    H : TYPE
        DESCRIPTION.
    edges : TYPE
        DESCRIPTION.
    runlength : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    vocab_counts : TYPE
        DESCRIPTION.

    '''
    
    
    vocab_counts = {'A':np.zeros((runlength+1)), 'B':np.zeros((runlength+1)), 'AB':np.zeros((runlength+1))}
    vocab_counts['A'][0] = H.count_by_attr('vocab', ['A'], False)
    vocab_counts['B'][0] = H.count_by_attr('vocab', ['B'], False)
    vocab_counts['AB'][0] = H.count_by_attr('vocab', ['A', 'B'], False)+H.count_by_attr('vocab', ['B', 'A'], False)
    #print(vocab_counts['AB'][0])
    H.add_edges_from(edges[0])
    random_edges = rand.choice(H.edges.members(), size = runlength)
    for i,edge in enumerate(random_edges):
        diff_dict = H.interact_and_advance(edge, verbose=verbose)
        vocab_counts['A'][i+1] = vocab_counts['A'][i] + diff_dict['A']
        vocab_counts['B'][i+1] = vocab_counts['B'][i] + diff_dict['B']
        vocab_counts['AB'][i+1] = vocab_counts['AB'][i] + diff_dict['AB']
        
    return vocab_counts

def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id

def run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, social_structure):
    edges, unique_id = get_edges_and_uniques(f'data/aggr_15min_cliques_thr2_{social_structure}.json')

    for k in tqdm(range(ensemble_size)):
        H = Hypergraph()

        number_committed = round(len(unique_id)*prop_committed)
        rand.shuffle(unique_id)

        committed_nodes, uncommitted_nodes = np.split(np.array(unique_id), [number_committed])
        
        H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=beta_non_committed)
        
        H.add_naming_game_node(committed_nodes, ['B'], True, beta=beta_committed)

        output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{run_length}_{ensemble_size}'
        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats = run_naming_game(H, edges, run_length, False)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])


#%%
