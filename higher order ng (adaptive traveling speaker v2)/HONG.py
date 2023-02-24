#%%

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
                    
                    self.move_speaker(speaker, edge_id)
                            
                        
                        
        
                        
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
            
        return diff_dict
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
        
        while new_edge_id not in self.nodes.memberships(speaker_id):
            new_edge_id = rand.choice(self.edges)
        self.remove_node_from_edge(edge_id, speaker_id)
        self.add_node_to_edge(new_edge_id, speaker_id)
        
        
        
        
        
        
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
        vocab_counts = {'A':np.zeros((runlength+1)), 'B':np.zeros((runlength+1)), 'AB':np.zeros((runlength+1))}
        vocab_counts['A'][0] = self.count_by_attr('vocab', ['A'], False)
        vocab_counts['B'][0] = self.count_by_attr('vocab', ['B'], False)
        vocab_counts['AB'][0] = self.count_by_attr('vocab', ['A', 'B'], False)+self.count_by_attr('vocab', ['B', 'A'], False)
        
        
        for i in range(runlength):
            diff_dict = self.interact_and_advance(verbose=verbose)
            vocab_counts['A'][i+1] = vocab_counts['A'][i] + diff_dict['A']
            vocab_counts['B'][i+1] = vocab_counts['B'][i] + diff_dict['B']
            vocab_counts['AB'][i+1] = vocab_counts['AB'][i] + diff_dict['AB']
            if vocab_counts['AB'][i+1] == 0 and vocab_counts['A'][i] == 0:
                vocab_counts['AB'][i+2:] = np.zeros((runlength-i-1))
                vocab_counts['A'][i+2:] = np.zeros((runlength-i-1))
                vocab_counts['B'][i+2:] = np.ones((runlength-i-1))
                vocab_counts['time_to_consensus'] = i+1
                break
        vocab_counts['time_to_consensus'] = vocab_counts.get('time_to_consensus', np.inf)
        return vocab_counts


def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id



def run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, social_structure, q, rule='Unanimous', thr=3):
    
    ### this line can be changed depending on which threshold we would like to use, 2 is our data, and data relating to other values come from https://github.com/iaciac/higher-order-NG
    edges, unique_id = get_edges_and_uniques(f'../data/aggr_15min_cliques_thr{thr}_{social_structure}.json')
    ###
    output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_q={q}_{run_length}_{ensemble_size}'
    
    ### This part deletes a file if it already exists
    if os.path.exists(f"outputs/{output_fname}.csv"):
        os.remove(f"outputs/{output_fname}.csv")
    ###

    for k in tqdm(range(ensemble_size)):
        H = HigherOrderNamingGame(rule=rule)

        number_committed = round(len(unique_id)*prop_committed)
        rand.shuffle(unique_id)

        committed_nodes, uncommitted_nodes = np.split(np.array(unique_id), [number_committed])
        
        H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=beta_non_committed, q = q)
        
        H.add_naming_game_node(committed_nodes, ['B'], True, beta=beta_committed, q = q)
        
        H.add_edges_from(edges[0])
        
        initial_size = H.edges.size.aslist()

        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats = H.run(edges,run_length, False)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])
            
        final_size = H.edges.size.aslist()
        
        with open(f'aux_outputs/{output_fname}.csv', 'a') as h:
            write = csv.writer(h)
            write.writerow(initial_size)
            write.writerow(final_size)
            h.close()
        







